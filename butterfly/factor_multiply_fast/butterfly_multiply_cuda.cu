#include <stdio.h>
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THCAtomics.cuh>  // For atomicAdd on Half
#include <c10/macros/Macros.h>  // For __launch_bounds__
#include <thrust/complex.h>
#include <thrust/pair.h>
#include <thrust/tuple.h>

#define thc_cos std::cos
#define thc_sin std::sin

#define FULL_MASK 0xffffffff

static constexpr int SMEM_PER_MP = 64 * (1 << 10);
static constexpr int MAX_SMEM_PER_BLOCK = 48 * (1 << 10);
// static constexpr int MAX_BLOCK_SIZE = 1024;
// static constexpr int WORK_PER_THREAD = 16;
// static constexpr int ELEMENTARY_SIZE = MAX_BLOCK_SIZE / 2;
// static constexpr int MAX_N_FACTORS = 10;
static constexpr int ITEMS_PER_THREAD_FORWARD = 4;
static constexpr int ITEMS_PER_THREAD_BACKWARD = 16;

template <typename T, size_t N>
using CudaAcsr = at::PackedTensorAccessor<T, N, at::RestrictPtrTraits, int32_t>;

constexpr __host__ __device__ int min_const(int x, int y) { return x <= y ? x : y; }
constexpr __host__ __device__ int min_const(int x, int y, int z) { return min_const(min_const(x, y), z); }
constexpr __host__ __device__ int max_const(int x, int y) { return x >= y ? x : y; }
constexpr __host__ __device__ int div_up_const(int a, int b) { return (a + b - 1) / b; }

// __host__ __device__ static inline int64_t div_up(int64_t a, int64_t b) {
//   return (a + b - 1) / b;
// }

__host__ __device__ static inline int div_up(int a, int b) {
  return (a + b - 1) / b;
}

template <typename scalar_t>
static __device__  __forceinline__
void atomicAdd(thrust::complex<scalar_t> *address,
               thrust::complex<scalar_t> val) {
  atomicAdd((scalar_t *)address, val.real());
  atomicAdd((scalar_t *)address + 1, val.imag());
}

template <typename scalar_t>
static __device__  __forceinline__
thrust::complex<scalar_t> __shfl_down_sync(unsigned int mask,
                                           thrust::complex<scalar_t> value,
                                           unsigned int delta,
                                           int width = warpSize) {
  return thrust::complex<scalar_t>(__shfl_down_sync(mask, value.real(), delta, width),
                                   __shfl_down_sync(mask, value.imag(), delta, width));
}

// 2x2 matrix [a, b; c, d] multiplied by a vector [x, y]
template <typename scalar_t>
static __device__  __forceinline__
thrust::pair<scalar_t, scalar_t> mult2x2(scalar_t a, scalar_t b, scalar_t c,
                                         scalar_t d, scalar_t x, scalar_t y) {
  return thrust::make_pair(a * x + b * y, c * x + d * y);
}

template <int items_per_thread, int smem_per_thread=items_per_thread, typename scalar_t>
__device__ __forceinline__ void block_exchange(scalar_t *temp_storage,
                                               scalar_t values[items_per_thread],
                                               int old_idx,
                                               int new_idx,
                                               int nthreads) {
  constexpr int nsteps = div_up_const(items_per_thread, smem_per_thread);
  #pragma unroll
  for (int i = 0; i < nsteps; i++) {
    if (i > 0) {
      __syncthreads();
    }
    #pragma unroll
    for (int item = 0; (item < smem_per_thread) && (i * smem_per_thread + item < items_per_thread); item++) {
      temp_storage[old_idx + item * nthreads] = values[i * smem_per_thread + item];
    }
    __syncthreads();
    #pragma unroll
    for (int item = 0; (item < smem_per_thread) && (i * smem_per_thread + item < items_per_thread); item++) {
      values[i * smem_per_thread + item] = temp_storage[new_idx + item * nthreads];
    }
  }
}

template <int nsteps, int items_per_thread, typename scalar_t>
__device__ __forceinline__ void b_untied_forward(const CudaAcsr<scalar_t, 4> twiddle_a,
                                                 scalar_t input_val[items_per_thread],
                                                 int log_min_stride,
                                                 int idx) {
  const int s = blockIdx.y + gridDim.y * blockIdx.z;  // For conv2d butterfly as well
  #pragma unroll
  for (int i = 0; i < nsteps; i++) {
    int lane_mask = 1 << i;
    int log_stride = i + log_min_stride;
    const scalar_t twiddle_val[2] = {twiddle_a[s][log_stride][0][idx],
                                     twiddle_a[s][log_stride][1][idx]};
    #pragma unroll
    for (int item = 0; item < items_per_thread; item++) {
      scalar_t input_val_other = __shfl_xor_sync(FULL_MASK, input_val[item], lane_mask);
      input_val[item] = twiddle_val[0] * input_val[item] + twiddle_val[1] * input_val_other;
    }
  }
}

template <int log_n, int items_per_thread, int min_blocks_per_mp=1, int max_smem_per_thread=items_per_thread,
            typename scalar_t, typename Function0, typename Function1>
C10_LAUNCH_BOUNDS_2(1 << log_n, min_blocks_per_mp)
__global__ void butterfly_multiply_untied_forward_fast_cuda_kernel(const CudaAcsr<scalar_t, 4> twiddle_a,
                                                                   Function0 load_input,
                                                                   Function1 save_output,
                                                                   int batch_size) {
  constexpr int n = 1 << log_n;
  constexpr int smem_limit = min_const(SMEM_PER_MP / min_blocks_per_mp, MAX_SMEM_PER_BLOCK);
  constexpr int smem_per_thread = min_const(max_smem_per_thread, items_per_thread,
                                            smem_limit / (n * sizeof(scalar_t)));
  __shared__ scalar_t temp_storage[n * smem_per_thread];
  scalar_t input_val[items_per_thread];
  load_input(input_val, threadIdx.x);
  b_untied_forward<min_const(log_n, 5), items_per_thread>(twiddle_a, input_val, 0, threadIdx.x);
  if (log_n > 5) {
    constexpr int log_nwarps = max_const(log_n - 5, 1);  // Take max to avoid compiler's warning
    // int new_idx = (threadIdx.x % (1 << log_nwarps)) * warpSize + threadIdx.x / (1 << log_nwarps);
    const int new_idx = (threadIdx.x & ((1 << log_nwarps) - 1)) * warpSize + (threadIdx.x >> log_nwarps);
    block_exchange<items_per_thread, smem_per_thread>(temp_storage, input_val, threadIdx.x, new_idx, n);
    b_untied_forward<log_n - 5, items_per_thread>(twiddle_a, input_val, 5, new_idx);
    // Don't need __syncthreads() before block_exchange because threads are writing to the same indices.
    block_exchange<items_per_thread, smem_per_thread>(temp_storage, input_val, new_idx, threadIdx.x, n);
  }
  save_output(input_val, threadIdx.x);
}

void butterfly_multiply_untied_forward_fast_cuda(const at::Tensor &twiddle,
                                                 const at::Tensor &input,
                                                 at::Tensor &output) {
  int batch_size = input.size(0);
  const int nstack = input.size(1);
  const int n = input.size(2);
  const int log_n = int(log2((double) n));
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "butterfly_multiply_untied_forward_fast_cuda", [&] {
    using accscalar_t = at::acc_type<scalar_t, true>;
    const auto twiddle_a = twiddle.packed_accessor<scalar_t, 4, at::RestrictPtrTraits, int32_t>();
    const auto input_a = input.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>();
    auto output_a = output.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>();
    dim3 block(n);
    dim3 grid(div_up(batch_size, ITEMS_PER_THREAD_FORWARD), 1, nstack);
    auto load_input = [batch_size, n, input_a] __device__ (scalar_t* input_val, int idx) {
      for (int i = idx; i < n; i += blockDim.x) {
        const int s = blockIdx.z;
        #pragma unroll
        for (int item = 0; item < ITEMS_PER_THREAD_FORWARD; item++){
          const int b = blockIdx.x * ITEMS_PER_THREAD_FORWARD + item;
          input_val[item] = b < batch_size ? input_a[b][s][i] : 0;
        }
      }
    };
    auto save_output = [batch_size, n, output_a] __device__ (scalar_t* output_val, int idx) mutable {
      for (int i = idx; i < n; i += blockDim.x) {
        const int s = blockIdx.z;
        #pragma unroll
        for (int item = 0; item < ITEMS_PER_THREAD_FORWARD; item++){
          const int b = blockIdx.x * ITEMS_PER_THREAD_FORWARD + item;
          if (b < batch_size) {
            output_a[b][s][i] = output_val[item];
          }
        }
      }
    };
    auto stream = at::cuda::getCurrentCUDAStream();
    switch (log_n)
      {
      case 1:
        butterfly_multiply_untied_forward_fast_cuda_kernel<1, ITEMS_PER_THREAD_FORWARD>
          <<<grid, block, 0, stream>>>(twiddle_a, load_input, save_output, batch_size); break;
      case 2:
        butterfly_multiply_untied_forward_fast_cuda_kernel<2, ITEMS_PER_THREAD_FORWARD>
          <<<grid, block, 0, stream>>>(twiddle_a, load_input, save_output, batch_size); break;
      case 3:
        butterfly_multiply_untied_forward_fast_cuda_kernel<3, ITEMS_PER_THREAD_FORWARD>
          <<<grid, block, 0, stream>>>(twiddle_a, load_input, save_output, batch_size); break;
      case 4:
        butterfly_multiply_untied_forward_fast_cuda_kernel<4, ITEMS_PER_THREAD_FORWARD>
          <<<grid, block, 0, stream>>>(twiddle_a, load_input, save_output, batch_size); break;
      case 5:
        butterfly_multiply_untied_forward_fast_cuda_kernel<5, ITEMS_PER_THREAD_FORWARD>
          <<<grid, block, 0, stream>>>(twiddle_a, load_input, save_output, batch_size); break;
      case 6:
        butterfly_multiply_untied_forward_fast_cuda_kernel<6, ITEMS_PER_THREAD_FORWARD>
          <<<grid, block, 0, stream>>>(twiddle_a, load_input, save_output, batch_size); break;
      case 7:
        butterfly_multiply_untied_forward_fast_cuda_kernel<7, ITEMS_PER_THREAD_FORWARD>
          <<<grid, block, 0, stream>>>(twiddle_a, load_input, save_output, batch_size); break;
      case 8:
        butterfly_multiply_untied_forward_fast_cuda_kernel<8, ITEMS_PER_THREAD_FORWARD>
          <<<grid, block, 0, stream>>>(twiddle_a, load_input, save_output, batch_size); break;
      case 9:
        butterfly_multiply_untied_forward_fast_cuda_kernel<9, ITEMS_PER_THREAD_FORWARD>
          <<<grid, block, 0, stream>>>(twiddle_a, load_input, save_output, batch_size); break;
      case 10:
        butterfly_multiply_untied_forward_fast_cuda_kernel<10, ITEMS_PER_THREAD_FORWARD>
          <<<grid, block, 0, stream>>>(twiddle_a, load_input, save_output, batch_size); break;
      }
  });
  AT_CHECK(cudaGetLastError() == cudaSuccess,
     "butterfly_multiply_untied_forward_fast_cuda failed with error code ",
     cudaGetLastError());
}

template <int nsteps, int items_per_thread,
            int reg_storage_per_thread=items_per_thread, typename scalar_t>
__device__ __forceinline__ void b_untied_forward_backward(const CudaAcsr<scalar_t, 4> twiddle_a,
                                                          CudaAcsr<scalar_t, 4> d_twiddle_a,
                                                          scalar_t input_val[items_per_thread],
                                                          scalar_t grad_val[items_per_thread],
                                                          int log_min_stride,
                                                          int id) {
  constexpr int nslices = div_up_const(items_per_thread, reg_storage_per_thread);
  const int s = blockIdx.y + gridDim.y * blockIdx.z;  // For conv2d butterfly as well
  scalar_t twiddle_val[nsteps][2];
  scalar_t d_twiddle_val[nsteps][2] = {0};
  scalar_t input_val_storage[nsteps][items_per_thread];
  #pragma unroll
  for (int i = 0; i < nslices; i++) {
    #pragma unroll
    for (int item = 0; (item < reg_storage_per_thread) && (i * reg_storage_per_thread + item < items_per_thread); item++) {
      input_val_storage[0][item] = input_val[i * reg_storage_per_thread + item];
    }
    #pragma unroll
    for (int step = 0; step < nsteps; step++) {
      int lane_mask = 1 << step;
      int log_stride = step + log_min_stride;
      if (i == 0) {
        twiddle_val[step][0] = twiddle_a[s][log_stride][0][id];
        twiddle_val[step][1] = twiddle_a[s][log_stride][1][id];
      }
      if (step < nsteps - 1) {  // Don't need input for the last step
        #pragma unroll
        for (int item = 0; (item < reg_storage_per_thread) && (i * reg_storage_per_thread + item < items_per_thread); item++) {
          scalar_t input_val_other = __shfl_xor_sync(FULL_MASK, input_val_storage[step][item], lane_mask);
          input_val_storage[step + 1][item] = twiddle_val[step][0] * input_val_storage[step][item]
            + twiddle_val[step][1] * input_val_other;
        }
      }
    }
    #pragma unroll
    for (int step = nsteps - 1; step >= 0; step--) {
      int lane_mask = 1 << step;
      int log_stride = step + log_min_stride;
      #pragma unroll
      for (int item = 0; (item < reg_storage_per_thread) && (i * reg_storage_per_thread + item < items_per_thread); item++) {
        int item_offset = i * reg_storage_per_thread + item;
        d_twiddle_val[step][0] += grad_val[item_offset] * input_val_storage[step][item];
        scalar_t input_val_other = __shfl_xor_sync(FULL_MASK, input_val_storage[step][item], lane_mask);
        d_twiddle_val[step][1] += grad_val[item_offset] * input_val_other;
        grad_val[item_offset] = twiddle_val[step][0] * grad_val[item_offset]
          + __shfl_xor_sync(FULL_MASK, twiddle_val[step][1] * grad_val[item_offset], lane_mask);
      }
      if (i == nslices - 1) {
        // if (id >= 9999) {
        // if (threadIdx.x < 128) {
        if (true) {
          atomicAdd(&d_twiddle_a[s][log_stride][0][id], d_twiddle_val[step][0]);
          atomicAdd(&d_twiddle_a[s][log_stride][1][id], d_twiddle_val[step][1]);
        }
      }
    }
  }
}

template <int log_n, int items_per_thread, int max_reg_storage_per_thread=items_per_thread,
            int min_blocks_per_mp=1, int max_smem_per_thread=items_per_thread,
            typename scalar_t, typename Function0, typename Function1, typename Function2>
C10_LAUNCH_BOUNDS_2(1 << log_n, min_blocks_per_mp)
__global__ void butterfly_multiply_untied_forward_backward_fast_cuda_kernel(const CudaAcsr<scalar_t, 4> twiddle_a,
                                                                            Function0 load_input,
                                                                            Function1 load_grad,
                                                                            CudaAcsr<scalar_t, 4> d_twiddle_a,
                                                                            Function2 save_d_input,
                                                                            int batch_size) {
  constexpr int n = 1 << log_n;
  constexpr int smem_limit = min_const(SMEM_PER_MP / min_blocks_per_mp, MAX_SMEM_PER_BLOCK);
  constexpr int smem_per_thread = min_const(max_smem_per_thread, items_per_thread,
                                            smem_limit / (n * sizeof(scalar_t)));
  constexpr int reg_storage_per_thread = min_const(max_reg_storage_per_thread, items_per_thread);
  __shared__ scalar_t temp_storage[n * smem_per_thread];
  scalar_t input_val[items_per_thread];
  scalar_t grad_val[items_per_thread];
  if (log_n > 5) {
    load_input(input_val, threadIdx.x);
    b_untied_forward<min_const(log_n, 5), items_per_thread>(twiddle_a, input_val, 0, threadIdx.x);
    constexpr int log_nwarps = max_const(log_n - 5, 1);  // Take max to avoid compiler's warning
    // const int new_idx = (threadIdx.x % (1 << log_nwarps)) * warpSize + threadIdx.x / (1 << log_nwarps);
    const int new_idx = (threadIdx.x & ((1 << log_nwarps) - 1)) * warpSize + (threadIdx.x >> log_nwarps);
    block_exchange<items_per_thread, smem_per_thread>(temp_storage, input_val, threadIdx.x, new_idx, n);
    load_grad(grad_val, new_idx);
    b_untied_forward_backward<log_nwarps, items_per_thread, reg_storage_per_thread>
      (twiddle_a, d_twiddle_a, input_val, grad_val, 5, new_idx);
    // Don't need __syncthreads() before block_exchange because threads are writing to the same indices.
    block_exchange<items_per_thread, smem_per_thread>(temp_storage, grad_val, new_idx, threadIdx.x, n);
  } else {
    load_grad(grad_val, threadIdx.x);
  }
  load_input(input_val, threadIdx.x);
  b_untied_forward_backward<min_const(log_n, 5), items_per_thread, reg_storage_per_thread>
    (twiddle_a, d_twiddle_a, input_val, grad_val, 0, threadIdx.x);
  save_d_input(grad_val, threadIdx.x);
}

void butterfly_multiply_untied_forward_backward_fast_cuda(const at::Tensor &twiddle,
                                                          const at::Tensor &input,
                                                          const at::Tensor &grad,
                                                          at::Tensor& d_twiddle,
                                                          at::Tensor& d_input) {
  int batch_size = input.size(0);
  const int nstack = input.size(1);
  const int n = input.size(2);
  const int log_n = int(log2((double) n));
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "butterfly_multiply_untied_forward_backward_fast_cuda", [&] {
    using accscalar_t = at::acc_type<scalar_t, true>;
    const auto twiddle_a = twiddle.packed_accessor<scalar_t, 4, at::RestrictPtrTraits, int32_t>();
    const auto input_a = input.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>();
    const auto grad_a = grad.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>();
    auto d_twiddle_a = d_twiddle.packed_accessor<scalar_t, 4, at::RestrictPtrTraits, int32_t>();
    auto d_input_a = d_input.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>();
    dim3 block(n);
    dim3 grid(div_up(batch_size, ITEMS_PER_THREAD_BACKWARD), 1, nstack);
    auto load_input = [batch_size, n, input_a] __device__ (scalar_t* input_val, int idx) {
      for (int i = idx; i < n; i += blockDim.x) {
        const int s = blockIdx.z;
        #pragma unroll
        for (int item = 0; item < ITEMS_PER_THREAD_BACKWARD; item++){
          const int b = blockIdx.x * ITEMS_PER_THREAD_BACKWARD + item;
          input_val[item] = b < batch_size ? input_a[b][s][i] : 0;
        }
      }
    };
    auto load_grad = [batch_size, n, grad_a] __device__ (scalar_t* grad_val, int idx) {
      for (int i = idx; i < n; i += blockDim.x) {
        const int s = blockIdx.z;
        #pragma unroll
        for (int item = 0; item < ITEMS_PER_THREAD_BACKWARD; item++){
          const int b = blockIdx.x * ITEMS_PER_THREAD_BACKWARD + item;
          grad_val[item] = b < batch_size ? grad_a[b][s][i] : 0;
        }
      }
    };
    auto save_d_input = [batch_size, n, d_input_a] __device__ (scalar_t* grad_val, int idx) mutable {
      for (int i = idx; i < n; i += blockDim.x) {
        const int s = blockIdx.z;
        #pragma unroll
        for (int item = 0; item < ITEMS_PER_THREAD_BACKWARD; item++){
          const int b = blockIdx.x * ITEMS_PER_THREAD_BACKWARD + item;
          if (b < batch_size) {
            d_input_a[b][s][i] = grad_val[item];
          }
        }
      }
    };
    auto stream = at::cuda::getCurrentCUDAStream();
    switch (log_n)
      {
      case 1:
        butterfly_multiply_untied_forward_backward_fast_cuda_kernel<1, ITEMS_PER_THREAD_BACKWARD>
          <<<grid, block, 0, stream>>>(twiddle_a, load_input, load_grad, d_twiddle_a, save_d_input, batch_size); break;
      case 2:
        butterfly_multiply_untied_forward_backward_fast_cuda_kernel<2, ITEMS_PER_THREAD_BACKWARD>
          <<<grid, block, 0, stream>>>(twiddle_a, load_input, load_grad, d_twiddle_a, save_d_input, batch_size); break;
      case 3:
        butterfly_multiply_untied_forward_backward_fast_cuda_kernel<3, ITEMS_PER_THREAD_BACKWARD>
          <<<grid, block, 0, stream>>>(twiddle_a, load_input, load_grad, d_twiddle_a, save_d_input, batch_size); break;
      case 4:
        butterfly_multiply_untied_forward_backward_fast_cuda_kernel<4, ITEMS_PER_THREAD_BACKWARD>
          <<<grid, block, 0, stream>>>(twiddle_a, load_input, load_grad, d_twiddle_a, save_d_input, batch_size); break;
      case 5:
        butterfly_multiply_untied_forward_backward_fast_cuda_kernel<5, ITEMS_PER_THREAD_BACKWARD>
          <<<grid, block, 0, stream>>>(twiddle_a, load_input, load_grad, d_twiddle_a, save_d_input, batch_size); break;
      case 6:
        butterfly_multiply_untied_forward_backward_fast_cuda_kernel<6, ITEMS_PER_THREAD_BACKWARD>
          <<<grid, block, 0, stream>>>(twiddle_a, load_input, load_grad, d_twiddle_a, save_d_input, batch_size); break;
      case 7:
        butterfly_multiply_untied_forward_backward_fast_cuda_kernel<7, ITEMS_PER_THREAD_BACKWARD>
          <<<grid, block, 0, stream>>>(twiddle_a, load_input, load_grad, d_twiddle_a, save_d_input, batch_size); break;
      case 8:
        butterfly_multiply_untied_forward_backward_fast_cuda_kernel<8, ITEMS_PER_THREAD_BACKWARD>
          <<<grid, block, 0, stream>>>(twiddle_a, load_input, load_grad, d_twiddle_a, save_d_input, batch_size); break;
      case 9:
        butterfly_multiply_untied_forward_backward_fast_cuda_kernel<9, ITEMS_PER_THREAD_BACKWARD>
          <<<grid, block, 0, stream>>>(twiddle_a, load_input, load_grad, d_twiddle_a, save_d_input, batch_size); break;
      case 10:
        butterfly_multiply_untied_forward_backward_fast_cuda_kernel<10, ITEMS_PER_THREAD_BACKWARD>
          <<<grid, block, 0, stream>>>(twiddle_a, load_input, load_grad, d_twiddle_a, save_d_input, batch_size); break;
      }
  });
  AT_CHECK(cudaGetLastError() == cudaSuccess,
     "butterfly_multiply_untied_forward_backward_fast_cuda failed with error code ",
     cudaGetLastError());
}