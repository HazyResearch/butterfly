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
static constexpr int ITEMS_PER_THREAD_FORWARD[14] = {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4};
static constexpr int ITEMS_PER_THREAD_BACKWARD[14] = {16, 16, 16, 16, 16, 16, 16, 16, 16, 4, 1, 1, 1, 1};

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

template<typename scalar_t>
struct InputReader {
  const CudaAcsr<scalar_t, 3> input_a;
  const int batch_size;
  const int n;
  InputReader(const at::Tensor input):
    input_a(input.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>()),
      batch_size(input.size(0)),
      n(input.size(2)) {}

  template<int items_per_thread, int mult_per_warp=1>
  __device__ __forceinline__ void load(scalar_t input_val[mult_per_warp][items_per_thread],
                                       int idx) {
    #pragma unroll
    for (int mult = 0; mult < mult_per_warp; mult++) {
      int i = mult * warpSize + idx;
      const int s = blockIdx.z;
      #pragma unroll
      for (int item = 0; item < items_per_thread; item++){
        const int b = blockIdx.x * items_per_thread + item;
        input_val[mult][item] = b < batch_size ? input_a[b][s][i] : 0;
      }
    }
  }

};

template<typename scalar_t>
struct OutputWriter {
  CudaAcsr<scalar_t, 3> output_a;
  const int batch_size;
  const int n;
  OutputWriter(at::Tensor output):
    output_a(output.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>()),
      batch_size(output.size(0)),
      n(output.size(2)) {}

  template<int items_per_thread, int mult_per_warp=1>
  __device__ __forceinline__ void save(scalar_t output_val[mult_per_warp][items_per_thread],
                                       int idx) {
    #pragma unroll
    for (int mult = 0; mult < mult_per_warp; mult++) {
      int i = mult * warpSize + idx;
      const int s = blockIdx.z;
      #pragma unroll
      for (int item = 0; item < items_per_thread; item++){
        const int b = blockIdx.x * items_per_thread + item;
        if (b < batch_size) {
          output_a[b][s][i] = output_val[mult][item];
        }
      }
    }
  }

};

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

template <int nsteps, bool increasing_stride, int items_per_thread,
            int mult_per_warp=1, typename scalar_t>
__device__ __forceinline__ void b_untied_forward(const CudaAcsr<scalar_t, 4> twiddle_a,
                                                 scalar_t input_val[mult_per_warp][items_per_thread],
                                                 int twiddle_idx_start,
                                                 int idx) {
  const int s = blockIdx.y + gridDim.y * blockIdx.z;  // For conv2d butterfly as well
  #pragma unroll
  for (int step = 0; step < nsteps; step++) {
    int log_stride = increasing_stride ? step : nsteps - 1 - step;
    int twiddle_idx = twiddle_idx_start + step;
    if (log_stride < 5) {
      int lane_mask = 1 << log_stride;
      #pragma unroll
      for (int mult = 0; mult < mult_per_warp; mult++) {
        // TODO: make num thread per warp an input argument
        const scalar_t twiddle_val[2] = {twiddle_a[s][twiddle_idx][0][mult * warpSize + idx],
                                         twiddle_a[s][twiddle_idx][1][mult * warpSize + idx]};
        #pragma unroll
        for (int item = 0; item < items_per_thread; item++) {
          scalar_t input_val_other = __shfl_xor_sync(FULL_MASK, input_val[mult][item], lane_mask);
          input_val[mult][item] = twiddle_val[0] * input_val[mult][item] + twiddle_val[1] * input_val_other;
        }
      }
    } else {
      int mult_stride = 1 << (log_stride - 5);
      #pragma unroll
      for (int i = 0; i < mult_per_warp / 2; i++) {
        int low_order_bits = i & (mult_stride - 1);  // int low_order_bits = i % mult_stride;
        int mult = 2 * (i - low_order_bits) + low_order_bits;
        const scalar_t twiddle_val[2][2]
          = {{twiddle_a[s][twiddle_idx][0][mult * warpSize + idx],
              twiddle_a[s][twiddle_idx][1][mult * warpSize + idx]},
             {twiddle_a[s][twiddle_idx][0][(mult + mult_stride) * warpSize + idx],
              twiddle_a[s][twiddle_idx][1][(mult + mult_stride) * warpSize + idx]}};
        #pragma unroll
        for (int item = 0; item < items_per_thread; item++) {
          scalar_t inputs[2] = {input_val[mult][item], input_val[mult + mult_stride][item]};
          input_val[mult][item] = twiddle_val[0][0] * inputs[0] + twiddle_val[0][1] * inputs[1];
          // The order of twiddle[1] is swapped by design
          input_val[mult + mult_stride][item] = twiddle_val[1][1] * inputs[0] + twiddle_val[1][0] * inputs[1];
        }
      }
    }
  }
}

template <int log_n, bool increasing_stride, int items_per_thread, int min_blocks_per_mp=1,
            int max_smem_per_thread=items_per_thread, typename scalar_t>
C10_LAUNCH_BOUNDS_2(1 << log_n, min_blocks_per_mp)
__global__ void butterfly_multiply_untied_forward_fast_cuda_kernel(const CudaAcsr<scalar_t, 4> twiddle_a,
                                                                   InputReader<scalar_t> input_reader,
                                                                   OutputWriter<scalar_t> output_writer,
                                                                   int batch_size) {
  constexpr int n = 1 << log_n;
  constexpr int smem_limit = min_const(SMEM_PER_MP / min_blocks_per_mp, MAX_SMEM_PER_BLOCK);
  constexpr int smem_per_thread = min_const(max_smem_per_thread, items_per_thread,
                                            smem_limit / (n * sizeof(scalar_t)));
  __shared__ scalar_t temp_storage[n * smem_per_thread];
  if (log_n == 6) {
    scalar_t input_val[2][items_per_thread];
    input_reader.load<items_per_thread, 2>(input_val, threadIdx.x);
    b_untied_forward<min_const(log_n, 6), increasing_stride, items_per_thread, 2>
    // b_untied_forward<min_const(log_n, 5), increasing_stride, items_per_thread, 2>
      (twiddle_a, input_val, 0, threadIdx.x);
    output_writer.save<items_per_thread, 2>(input_val, threadIdx.x);
    return;
  }
  scalar_t input_val[1][items_per_thread];
  input_reader.load<items_per_thread>(input_val, threadIdx.x);
  if (log_n <= 5) {
    b_untied_forward<min_const(log_n, 5), increasing_stride, items_per_thread>
      (twiddle_a, input_val, 0, threadIdx.x);
  } else {
    constexpr int log_nwarps = max_const(log_n - 5, 1);  // Take max to avoid compiler's warning
    // int new_idx = (threadIdx.x % (1 << log_nwarps)) * warpSize + threadIdx.x / (1 << log_nwarps);
    const int new_idx = (threadIdx.x & ((1 << log_nwarps) - 1)) * warpSize + (threadIdx.x >> log_nwarps);
    if (increasing_stride) {
      b_untied_forward<min_const(log_n, 5), true, items_per_thread>(twiddle_a, input_val, 0, threadIdx.x);
      block_exchange<items_per_thread, smem_per_thread>(temp_storage, input_val[0], threadIdx.x, new_idx, n);
      b_untied_forward<log_n - 5, true, items_per_thread>(twiddle_a, input_val, 5, new_idx);
      // Don't need __syncthreads() before block_exchange because threads are writing to the same indices.
      block_exchange<items_per_thread, smem_per_thread>(temp_storage, input_val[0], new_idx, threadIdx.x, n);
    } else {
      block_exchange<items_per_thread, smem_per_thread>(temp_storage, input_val[0], threadIdx.x, new_idx, n);
      b_untied_forward<log_n - 5, false, items_per_thread>(twiddle_a, input_val, 0, new_idx);
      block_exchange<items_per_thread, smem_per_thread>(temp_storage, input_val[0], new_idx, threadIdx.x, n);
      b_untied_forward<min_const(log_n, 5), false, items_per_thread>(twiddle_a, input_val, log_n - 5, threadIdx.x);
    }
  }
  output_writer.save<items_per_thread>(input_val, threadIdx.x);
}

void butterfly_multiply_untied_forward_fast_cuda(const at::Tensor &twiddle,
                                                 const at::Tensor &input,
                                                 at::Tensor &output,
                                                 bool increasing_stride) {
  int batch_size = input.size(0);
  const int nstack = input.size(1);
  const int n = input.size(2);
  const int log_n = int(log2((double) n));
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "butterfly_multiply_untied_forward_fast_cuda", [&] {
    using accscalar_t = at::acc_type<scalar_t, true>;
    const auto twiddle_a = twiddle.packed_accessor<scalar_t, 4, at::RestrictPtrTraits, int32_t>();
    const InputReader<scalar_t> input_reader(input);
    OutputWriter<scalar_t> output_writer(output);
    // dim3 block(n);
    dim3 block(log_n == 6 ? 32 : n);
    dim3 grid(div_up(batch_size, ITEMS_PER_THREAD_FORWARD[log_n - 1]), 1, nstack);
    auto stream = at::cuda::getCurrentCUDAStream();
    switch (log_n)
      {
      case 1:
        increasing_stride ? butterfly_multiply_untied_forward_fast_cuda_kernel<1, true, ITEMS_PER_THREAD_FORWARD[0]>
          <<<grid, block, 0, stream>>>(twiddle_a, input_reader, output_writer, batch_size)
          : butterfly_multiply_untied_forward_fast_cuda_kernel<1, false, ITEMS_PER_THREAD_FORWARD[0]>
          <<<grid, block, 0, stream>>>(twiddle_a, input_reader, output_writer, batch_size); break;
      case 2:
        increasing_stride ? butterfly_multiply_untied_forward_fast_cuda_kernel<2, true, ITEMS_PER_THREAD_FORWARD[1]>
          <<<grid, block, 0, stream>>>(twiddle_a, input_reader, output_writer, batch_size)
          : butterfly_multiply_untied_forward_fast_cuda_kernel<2, false, ITEMS_PER_THREAD_FORWARD[1]>
          <<<grid, block, 0, stream>>>(twiddle_a, input_reader, output_writer, batch_size); break;
      case 3:
        increasing_stride ? butterfly_multiply_untied_forward_fast_cuda_kernel<3, true, ITEMS_PER_THREAD_FORWARD[2]>
          <<<grid, block, 0, stream>>>(twiddle_a, input_reader, output_writer, batch_size)
          : butterfly_multiply_untied_forward_fast_cuda_kernel<3, false, ITEMS_PER_THREAD_FORWARD[2]>
          <<<grid, block, 0, stream>>>(twiddle_a, input_reader, output_writer, batch_size); break;
      case 4:
        increasing_stride ? butterfly_multiply_untied_forward_fast_cuda_kernel<4, true, ITEMS_PER_THREAD_FORWARD[3]>
          <<<grid, block, 0, stream>>>(twiddle_a, input_reader, output_writer, batch_size)
          : butterfly_multiply_untied_forward_fast_cuda_kernel<4, false, ITEMS_PER_THREAD_FORWARD[3]>
          <<<grid, block, 0, stream>>>(twiddle_a, input_reader, output_writer, batch_size); break;
      case 5:
        increasing_stride ? butterfly_multiply_untied_forward_fast_cuda_kernel<5, true, ITEMS_PER_THREAD_FORWARD[4]>
          <<<grid, block, 0, stream>>>(twiddle_a, input_reader, output_writer, batch_size)
          : butterfly_multiply_untied_forward_fast_cuda_kernel<5, false, ITEMS_PER_THREAD_FORWARD[4]>
          <<<grid, block, 0, stream>>>(twiddle_a, input_reader, output_writer, batch_size); break;
      case 6:
        increasing_stride ? butterfly_multiply_untied_forward_fast_cuda_kernel<6, true, ITEMS_PER_THREAD_FORWARD[5]>
          <<<grid, block, 0, stream>>>(twiddle_a, input_reader, output_writer, batch_size)
          : butterfly_multiply_untied_forward_fast_cuda_kernel<6, false, ITEMS_PER_THREAD_FORWARD[5]>
          <<<grid, block, 0, stream>>>(twiddle_a, input_reader, output_writer, batch_size); break;
      case 7:
        increasing_stride ? butterfly_multiply_untied_forward_fast_cuda_kernel<7, true, ITEMS_PER_THREAD_FORWARD[6]>
          <<<grid, block, 0, stream>>>(twiddle_a, input_reader, output_writer, batch_size)
          : butterfly_multiply_untied_forward_fast_cuda_kernel<7, false, ITEMS_PER_THREAD_FORWARD[6]>
          <<<grid, block, 0, stream>>>(twiddle_a, input_reader, output_writer, batch_size); break;
      case 8:
        increasing_stride ? butterfly_multiply_untied_forward_fast_cuda_kernel<8, true, ITEMS_PER_THREAD_FORWARD[7]>
          <<<grid, block, 0, stream>>>(twiddle_a, input_reader, output_writer, batch_size)
          : butterfly_multiply_untied_forward_fast_cuda_kernel<8, false, ITEMS_PER_THREAD_FORWARD[7]>
          <<<grid, block, 0, stream>>>(twiddle_a, input_reader, output_writer, batch_size); break;
      case 9:
        increasing_stride ? butterfly_multiply_untied_forward_fast_cuda_kernel<9, true, ITEMS_PER_THREAD_FORWARD[8]>
          <<<grid, block, 0, stream>>>(twiddle_a, input_reader, output_writer, batch_size)
          : butterfly_multiply_untied_forward_fast_cuda_kernel<9, false, ITEMS_PER_THREAD_FORWARD[8]>
          <<<grid, block, 0, stream>>>(twiddle_a, input_reader, output_writer, batch_size); break;
      case 10:
        increasing_stride ? butterfly_multiply_untied_forward_fast_cuda_kernel<10, true, ITEMS_PER_THREAD_FORWARD[9]>
          <<<grid, block, 0, stream>>>(twiddle_a, input_reader, output_writer, batch_size)
          : butterfly_multiply_untied_forward_fast_cuda_kernel<10, false, ITEMS_PER_THREAD_FORWARD[9]>
          <<<grid, block, 0, stream>>>(twiddle_a, input_reader, output_writer, batch_size); break;
      }
  });
  AT_CHECK(cudaGetLastError() == cudaSuccess,
     "butterfly_multiply_untied_forward_fast_cuda failed with error code ",
     cudaGetLastError());
}

template <int nsteps, bool increasing_stride, int items_per_thread,
            int reg_storage_per_thread=items_per_thread, typename scalar_t>
__device__ __forceinline__ void b_untied_forward_backward(const CudaAcsr<scalar_t, 4> twiddle_a,
                                                          CudaAcsr<scalar_t, 4> d_twiddle_a,
                                                          scalar_t input_val[items_per_thread],
                                                          scalar_t grad_val[items_per_thread],
                                                          int twiddle_idx_start,
                                                          int idx) {
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
      int lane_mask = increasing_stride ? (1 << step) : (1 << (nsteps - 1 - step));
      int twiddle_idx = step + twiddle_idx_start;
      if (i == 0) {
        twiddle_val[step][0] = twiddle_a[s][twiddle_idx][0][idx];
        twiddle_val[step][1] = twiddle_a[s][twiddle_idx][1][idx];
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
      int lane_mask = increasing_stride ? (1 << step) : (1 << (nsteps - 1 - step));
      int twiddle_idx = step + twiddle_idx_start;
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
        // if (idx >= 9999) {
        // if (threadIdx.x < 128) {
        if (true) {
          atomicAdd(&d_twiddle_a[s][twiddle_idx][0][idx], d_twiddle_val[step][0]);
          atomicAdd(&d_twiddle_a[s][twiddle_idx][1][idx], d_twiddle_val[step][1]);
        }
      }
    }
  }
}

template <int log_n, bool increasing_stride, int items_per_thread,
            int max_reg_storage_per_thread=items_per_thread,
            int min_blocks_per_mp=1, int max_smem_per_thread=items_per_thread,
            typename scalar_t>
C10_LAUNCH_BOUNDS_2(1 << log_n, min_blocks_per_mp)
__global__ void butterfly_multiply_untied_forward_backward_fast_cuda_kernel(const CudaAcsr<scalar_t, 4> twiddle_a,
                                                                            InputReader<scalar_t> input_reader,
                                                                            InputReader<scalar_t> grad_reader,
                                                                            CudaAcsr<scalar_t, 4> d_twiddle_a,
                                                                            OutputWriter<scalar_t> d_input_writer,
                                                                            int batch_size) {
  constexpr int n = 1 << log_n;
  constexpr int smem_limit = min_const(SMEM_PER_MP / min_blocks_per_mp, MAX_SMEM_PER_BLOCK);
  constexpr int smem_per_thread = min_const(max_smem_per_thread, items_per_thread,
                                            smem_limit / (n * sizeof(scalar_t)));
  constexpr int reg_storage_per_thread = min_const(max_reg_storage_per_thread, items_per_thread);
  __shared__ scalar_t temp_storage[n * smem_per_thread];
  scalar_t input_val[1][items_per_thread];
  scalar_t grad_val[1][items_per_thread];
  input_reader.load<items_per_thread>(input_val, threadIdx.x);
  if (log_n <= 5) {
    grad_reader.load<items_per_thread>(grad_val, threadIdx.x);
    b_untied_forward_backward<min_const(log_n, 5), increasing_stride, items_per_thread, reg_storage_per_thread>
      (twiddle_a, d_twiddle_a, input_val[0], grad_val[0], 0, threadIdx.x);
  } else {
    constexpr int log_nwarps = max_const(log_n - 5, 1);  // Take max to avoid compiler's warning
    // const int new_idx = (threadIdx.x % (1 << log_nwarps)) * warpSize + threadIdx.x / (1 << log_nwarps);
    const int new_idx = (threadIdx.x & ((1 << log_nwarps) - 1)) * warpSize + (threadIdx.x >> log_nwarps);
    if (increasing_stride) {
      b_untied_forward<min_const(log_n, 5), true, items_per_thread>(twiddle_a, input_val, 0, threadIdx.x);
      block_exchange<items_per_thread, smem_per_thread>(temp_storage, input_val[0], threadIdx.x, new_idx, n);
      grad_reader.load<items_per_thread>(grad_val, new_idx);
      b_untied_forward_backward<log_nwarps, true, items_per_thread, reg_storage_per_thread>
        (twiddle_a, d_twiddle_a, input_val[0], grad_val[0], 5, new_idx);
      // Don't need __syncthreads() before block_exchange because threads are writing to the same indices.
      block_exchange<items_per_thread, smem_per_thread>(temp_storage, grad_val[0], new_idx, threadIdx.x, n);
      input_reader.load<items_per_thread>(input_val, threadIdx.x);
      b_untied_forward_backward<min_const(log_n, 5), true, items_per_thread, reg_storage_per_thread>
        (twiddle_a, d_twiddle_a, input_val[0], grad_val[0], 0, threadIdx.x);
    } else {
      block_exchange<items_per_thread, smem_per_thread>(temp_storage, input_val[0], threadIdx.x, new_idx, n);
      b_untied_forward<log_nwarps, false, items_per_thread>(twiddle_a, input_val, 0, new_idx);
      block_exchange<items_per_thread, smem_per_thread>(temp_storage, input_val[0], new_idx, threadIdx.x, n);
      grad_reader.load<items_per_thread>(grad_val, threadIdx.x);
      b_untied_forward_backward<min_const(log_n, 5), false, items_per_thread, reg_storage_per_thread>
        (twiddle_a, d_twiddle_a, input_val[0], grad_val[0], log_nwarps, threadIdx.x);
      block_exchange<items_per_thread, smem_per_thread>(temp_storage, grad_val[0], threadIdx.x, new_idx, n);
      input_reader.load<items_per_thread>(input_val, new_idx);
      b_untied_forward_backward<log_nwarps, false, items_per_thread, reg_storage_per_thread>
        (twiddle_a, d_twiddle_a, input_val[0], grad_val[0], 0, new_idx);
      block_exchange<items_per_thread, smem_per_thread>(temp_storage, grad_val[0], new_idx, threadIdx.x, n);
    }
  }
  d_input_writer.save<items_per_thread>(grad_val, threadIdx.x);
}

void butterfly_multiply_untied_forward_backward_fast_cuda(const at::Tensor &twiddle,
                                                          const at::Tensor &input,
                                                          const at::Tensor &grad,
                                                          at::Tensor& d_twiddle,
                                                          at::Tensor& d_input,
                                                          bool increasing_stride) {
  int batch_size = input.size(0);
  const int nstack = input.size(1);
  const int n = input.size(2);
  const int log_n = int(log2((double) n));
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "butterfly_multiply_untied_forward_backward_fast_cuda", [&] {
    using accscalar_t = at::acc_type<scalar_t, true>;
    const auto twiddle_a = twiddle.packed_accessor<scalar_t, 4, at::RestrictPtrTraits, int32_t>();
    const InputReader<scalar_t> input_reader(input);
    const InputReader<scalar_t> grad_reader(grad);
    auto d_twiddle_a = d_twiddle.packed_accessor<scalar_t, 4, at::RestrictPtrTraits, int32_t>();
    OutputWriter<scalar_t> d_input_writer(d_input);
    dim3 block(n);
    dim3 grid(div_up(batch_size, ITEMS_PER_THREAD_BACKWARD[log_n - 1]), 1, nstack);
    auto stream = at::cuda::getCurrentCUDAStream();
    switch (log_n)
      {
      case 1:
        increasing_stride ? butterfly_multiply_untied_forward_backward_fast_cuda_kernel<1, true, ITEMS_PER_THREAD_BACKWARD[0]>
          <<<grid, block, 0, stream>>>(twiddle_a, input_reader, grad_reader, d_twiddle_a, d_input_writer, batch_size)
          : butterfly_multiply_untied_forward_backward_fast_cuda_kernel<1, false, ITEMS_PER_THREAD_BACKWARD[0]>
          <<<grid, block, 0, stream>>>(twiddle_a, input_reader, grad_reader, d_twiddle_a, d_input_writer, batch_size); break;
      case 2:
        increasing_stride ? butterfly_multiply_untied_forward_backward_fast_cuda_kernel<2, true, ITEMS_PER_THREAD_BACKWARD[1]>
          <<<grid, block, 0, stream>>>(twiddle_a, input_reader, grad_reader, d_twiddle_a, d_input_writer, batch_size)
          : butterfly_multiply_untied_forward_backward_fast_cuda_kernel<2, false, ITEMS_PER_THREAD_BACKWARD[1]>
          <<<grid, block, 0, stream>>>(twiddle_a, input_reader, grad_reader, d_twiddle_a, d_input_writer, batch_size); break;
      case 3:
        increasing_stride ? butterfly_multiply_untied_forward_backward_fast_cuda_kernel<3, true, ITEMS_PER_THREAD_BACKWARD[2]>
          <<<grid, block, 0, stream>>>(twiddle_a, input_reader, grad_reader, d_twiddle_a, d_input_writer, batch_size)
          : butterfly_multiply_untied_forward_backward_fast_cuda_kernel<3, false, ITEMS_PER_THREAD_BACKWARD[2]>
          <<<grid, block, 0, stream>>>(twiddle_a, input_reader, grad_reader, d_twiddle_a, d_input_writer, batch_size); break;
      case 4:
        increasing_stride ? butterfly_multiply_untied_forward_backward_fast_cuda_kernel<4, true, ITEMS_PER_THREAD_BACKWARD[3]>
          <<<grid, block, 0, stream>>>(twiddle_a, input_reader, grad_reader, d_twiddle_a, d_input_writer, batch_size)
          : butterfly_multiply_untied_forward_backward_fast_cuda_kernel<4, false, ITEMS_PER_THREAD_BACKWARD[3]>
          <<<grid, block, 0, stream>>>(twiddle_a, input_reader, grad_reader, d_twiddle_a, d_input_writer, batch_size); break;
      case 5:
        increasing_stride ? butterfly_multiply_untied_forward_backward_fast_cuda_kernel<5, true, ITEMS_PER_THREAD_BACKWARD[4]>
          <<<grid, block, 0, stream>>>(twiddle_a, input_reader, grad_reader, d_twiddle_a, d_input_writer, batch_size)
          : butterfly_multiply_untied_forward_backward_fast_cuda_kernel<5, false, ITEMS_PER_THREAD_BACKWARD[4]>
          <<<grid, block, 0, stream>>>(twiddle_a, input_reader, grad_reader, d_twiddle_a, d_input_writer, batch_size); break;
      case 6:
        increasing_stride ? butterfly_multiply_untied_forward_backward_fast_cuda_kernel<6, true, ITEMS_PER_THREAD_BACKWARD[5]>
          <<<grid, block, 0, stream>>>(twiddle_a, input_reader, grad_reader, d_twiddle_a, d_input_writer, batch_size)
          : butterfly_multiply_untied_forward_backward_fast_cuda_kernel<6, false, ITEMS_PER_THREAD_BACKWARD[5]>
          <<<grid, block, 0, stream>>>(twiddle_a, input_reader, grad_reader, d_twiddle_a, d_input_writer, batch_size); break;
      case 7:
        increasing_stride ? butterfly_multiply_untied_forward_backward_fast_cuda_kernel<7, true, ITEMS_PER_THREAD_BACKWARD[6]>
          <<<grid, block, 0, stream>>>(twiddle_a, input_reader, grad_reader, d_twiddle_a, d_input_writer, batch_size)
          : butterfly_multiply_untied_forward_backward_fast_cuda_kernel<7, false, ITEMS_PER_THREAD_BACKWARD[6]>
          <<<grid, block, 0, stream>>>(twiddle_a, input_reader, grad_reader, d_twiddle_a, d_input_writer, batch_size); break;
      case 8:
        increasing_stride ? butterfly_multiply_untied_forward_backward_fast_cuda_kernel<8, true, ITEMS_PER_THREAD_BACKWARD[7]>
          <<<grid, block, 0, stream>>>(twiddle_a, input_reader, grad_reader, d_twiddle_a, d_input_writer, batch_size)
          : butterfly_multiply_untied_forward_backward_fast_cuda_kernel<8, false, ITEMS_PER_THREAD_BACKWARD[7]>
          <<<grid, block, 0, stream>>>(twiddle_a, input_reader, grad_reader, d_twiddle_a, d_input_writer, batch_size); break;
      case 9:
        increasing_stride ? butterfly_multiply_untied_forward_backward_fast_cuda_kernel<9, true, ITEMS_PER_THREAD_BACKWARD[8]>
          <<<grid, block, 0, stream>>>(twiddle_a, input_reader, grad_reader, d_twiddle_a, d_input_writer, batch_size)
          : butterfly_multiply_untied_forward_backward_fast_cuda_kernel<9, false, ITEMS_PER_THREAD_BACKWARD[8]>
          <<<grid, block, 0, stream>>>(twiddle_a, input_reader, grad_reader, d_twiddle_a, d_input_writer, batch_size); break;
      case 10:
        increasing_stride ? butterfly_multiply_untied_forward_backward_fast_cuda_kernel<10, true, ITEMS_PER_THREAD_BACKWARD[9]>
          <<<grid, block, 0, stream>>>(twiddle_a, input_reader, grad_reader, d_twiddle_a, d_input_writer, batch_size)
          : butterfly_multiply_untied_forward_backward_fast_cuda_kernel<10, false, ITEMS_PER_THREAD_BACKWARD[9]>
          <<<grid, block, 0, stream>>>(twiddle_a, input_reader, grad_reader, d_twiddle_a, d_input_writer, batch_size); break;
      }
  });
  AT_CHECK(cudaGetLastError() == cudaSuccess,
     "butterfly_multiply_untied_forward_backward_fast_cuda failed with error code ",
     cudaGetLastError());
}