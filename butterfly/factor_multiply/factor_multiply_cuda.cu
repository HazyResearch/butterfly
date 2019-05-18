#include <stdio.h>
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THCAtomics.cuh>  // For atomicAdd on Half
#include <thrust/complex.h>
#include <thrust/pair.h>
#include <thrust/tuple.h>

// #define thc_cos THCNumerics<scalar_t>::cos
// #define thc_sin THCNumerics<scalar_t>::sin
#define thc_cos std::cos
#define thc_sin std::sin

#define FULL_MASK 0xffffffff

static constexpr int MAX_BLOCK_SIZE = 1024;
static constexpr int WORK_PER_THREAD = 16;
static constexpr int ELEMENTARY_SIZE = MAX_BLOCK_SIZE / 2;
static constexpr int MAX_N_FACTORS = 10;

template <typename T, size_t N>
using CudaAcsr = at::PackedTensorAccessor<T, N, at::RestrictPtrTraits, int32_t>;

__host__ __device__ static inline int64_t div_up(int64_t a, int64_t b) {
  return (a + b - 1) / b;
}

__host__ __device__ static inline int div_up(int a, int b) {
  return (a + b - 1) / b;
}

template <typename scalar_t>
static __device__  __forceinline__  void atomicAdd(thrust::complex<scalar_t> *address, thrust::complex<scalar_t> val) {
  atomicAdd((scalar_t *)address, val.real());
  atomicAdd((scalar_t *)address + 1, val.imag());
}

template <typename scalar_t>
static __device__  __forceinline__  thrust::complex<scalar_t> __shfl_down_sync(unsigned int mask, thrust::complex<scalar_t> value, unsigned int delta, int width = warpSize) {
  return thrust::complex<scalar_t>(__shfl_down_sync(mask, value.real(), delta, width),
                                   __shfl_down_sync(mask, value.imag(), delta, width));
}

// 2x2 matrix [a, b; c, d] multiplied by a vector [x, y]
template <typename scalar_t>
static __device__  __forceinline__  thrust::pair<scalar_t, scalar_t> mult2x2(scalar_t a, scalar_t b,
                                                                             scalar_t c, scalar_t d,
                                                                             scalar_t x, scalar_t y) {
  return thrust::make_pair(a * x + b * y, c * x + d * y);
}

template <typename scalar_t>
__global__ void butterfly_factor_multiply_cuda_kernel(const at::PackedTensorAccessor<scalar_t, 3> twiddle_a,
                                                      const at::PackedTensorAccessor<scalar_t, 3> input_a,
                                                      at::PackedTensorAccessor<scalar_t, 3> output_a) {
  const auto batch_size = input_a.size(0);
  const auto n = input_a.size(2);
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    const scalar_t twiddle_val[2][2] = {{twiddle_a[0][0][i], twiddle_a[0][1][i]},
                                        {twiddle_a[1][0][i], twiddle_a[1][1][i]}};
    for (int64_t b = blockIdx.y * blockDim.y + threadIdx.y; b < batch_size; b += blockDim.y * gridDim.y) {
      const scalar_t input_val[2] = {input_a[b][0][i], input_a[b][1][i]};
      #pragma unroll
      for (int j = 0; j <= 1; ++j) {
        output_a[b][j][i] = twiddle_val[j][0] * input_val[0] + twiddle_val[j][1] * input_val[1];
      }
    }
  }
}

template <typename scalar_t>
__global__ void butterfly_factor_multiply_complex_cuda_kernel(const at::PackedTensorAccessor<scalar_t, 4> twiddle_a,
                                                              const at::PackedTensorAccessor<scalar_t, 4> input_a,
                                                              at::PackedTensorAccessor<scalar_t, 4> output_a) {
  const auto batch_size = input_a.size(0);
  const auto n = input_a.size(2);
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    const scalar_t twiddle_val[2][2][2] = {{{twiddle_a[0][0][i][0], twiddle_a[0][0][i][1]},
                                            {twiddle_a[0][1][i][0], twiddle_a[0][1][i][1]}},
                                            {{twiddle_a[1][0][i][0], twiddle_a[1][0][i][1]},
                                            {twiddle_a[1][1][i][0], twiddle_a[1][1][i][1]}}};
    for (int64_t b = blockIdx.y * blockDim.y + threadIdx.y; b < batch_size; b += blockDim.y * gridDim.y) {
      const scalar_t input_val[2][2] = {{input_a[b][0][i][0], input_a[b][0][i][1]},
                                        {input_a[b][1][i][0], input_a[b][1][i][1]}};
      #pragma unroll
      for (int j = 0; j <= 1; ++j) {
        output_a[b][j][i][0] = twiddle_val[j][0][0] * input_val[0][0] - twiddle_val[j][0][1] * input_val[0][1]
          + twiddle_val[j][1][0] * input_val[1][0] - twiddle_val[j][1][1] * input_val[1][1];
        output_a[b][j][i][1] = twiddle_val[j][0][0] * input_val[0][1] + twiddle_val[j][0][1] * input_val[0][0]
          + twiddle_val[j][1][0] * input_val[1][1] + twiddle_val[j][1][1] * input_val[1][0];
      }
    }
  }
}

void butterfly_factor_multiply_cuda(const at::Tensor& twiddle, const at::Tensor& input, at::Tensor& output) {
  const auto batch_size = input.size(0);
  const auto n = input.size(2);
  dim3 block;
  block.x = std::min<int64_t>(MAX_BLOCK_SIZE, n);
  block.y = div_up(MAX_BLOCK_SIZE, block.x);
  dim3 grid(div_up(n, block.x), div_up(batch_size, block.y * WORK_PER_THREAD));
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "butterfly_factor_multiply_cuda", [&] {
    switch (input.dim()) {
      case 3:  // real
        {
          const auto twiddle_a = twiddle.packed_accessor<scalar_t, 3>();
          const auto input_a = input.packed_accessor<scalar_t, 3>();
          auto output_a = output.packed_accessor<scalar_t, 3>();
          butterfly_factor_multiply_cuda_kernel<scalar_t>
            <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, input_a, output_a);
          break;
        }
      case 4:  // complex
        {
          const auto twiddle_a = twiddle.packed_accessor<scalar_t, 4>();
          const auto input_a = input.packed_accessor<scalar_t, 4>();
          auto output_a = output.packed_accessor<scalar_t, 4>();
          butterfly_factor_multiply_complex_cuda_kernel<scalar_t>
            <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, input_a, output_a);
          break;
        }
      default:
        AT_ERROR("butterfly_factor_multiply requires input dimension 3 or 4");
    }
  });
  AT_CHECK(cudaGetLastError() == cudaSuccess,
     "butterfly_factor_multiply_cuda failed with error code ",
     cudaGetLastError());
}

template <typename T>
__device__ __forceinline__ T sum_strided(T val, T *temp, int stride, int len, int thread_id) {
  if (stride >= len) {
    return val;
  }
  // Warp reduction
  for (int offset = warpSize / 2; offset >= stride; offset /= 2) {
    val += __shfl_down_sync(FULL_MASK, val, offset);
  }
  // Block reduction
  int block_reduction_stride = max(warpSize, stride);
  int n_block_reductions = div_up(len, block_reduction_stride);
  __syncthreads();  // Otherwise previous reads might be wrong
  if (thread_id < len) {
    temp[(thread_id % block_reduction_stride) * n_block_reductions + (thread_id / block_reduction_stride)] = val;
  }
  __syncthreads();
  if (thread_id < n_block_reductions * stride) {
    val = temp[thread_id];
    for (int offset = n_block_reductions / 2; offset > 0; offset /= 2) {
      val += __shfl_down_sync(FULL_MASK, val, offset);
    }
  }
  return val;
}

template <typename scalar_t>
__global__ void butterfly_factor_multiply_backward_cuda_kernel(const at::PackedTensorAccessor<scalar_t, 3> grad_a,
                                                               const at::PackedTensorAccessor<scalar_t, 3> twiddle_a,
                                                               const at::PackedTensorAccessor<scalar_t, 3> input_a,
                                                               // at::PackedTensorAccessor<scalar_t, 4> d_twiddle_expanded_a,
                                                               at::PackedTensorAccessor<scalar_t, 3> d_twiddle_expanded_a,
                                                               at::PackedTensorAccessor<scalar_t, 3> d_input_a) {
  const int batch_size = input_a.size(0);
  const int n = input_a.size(2);
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    const scalar_t twiddle_val[2][2] = {{twiddle_a[0][0][i], twiddle_a[0][1][i]},
                                        {twiddle_a[1][0][i], twiddle_a[1][1][i]}};
    scalar_t d_twiddle_val[2][2] = {{0, 0}, {0, 0}};
    const int b_start = blockIdx.y * blockDim.y + threadIdx.y;
    // for (int64_t b = blockIdx.y * blockDim.y + threadIdx.y; b < batch_size; b += blockDim.y * gridDim.y) {
    for (int b = b_start; b < batch_size; b += blockDim.y * gridDim.y) {
      const scalar_t input_val[2] = {input_a[b][0][i], input_a[b][1][i]};
      const scalar_t grad_val[2] = {grad_a[b][0][i], grad_a[b][1][i]};
      #pragma unroll
      for (int j = 0; j <= 1; ++j) {
        // d_twiddle_expanded_a[b][j][0][i] = grad_val[j] * input_val[0];
        // d_twiddle_expanded_a[b][j][1][i] = grad_val[j] * input_val[1];
        // atomicAdd(&d_twiddle_expanded_a[j][0][i], grad_val[j] * input_val[0]);
        // atomicAdd(&d_twiddle_expanded_a[j][1][i], grad_val[j] * input_val[1]);
        d_twiddle_val[j][0] += grad_val[j] * input_val[0];
        d_twiddle_val[j][1] += grad_val[j] * input_val[1];
        d_input_a[b][j][i] = twiddle_val[0][j] * grad_val[0] + twiddle_val[1][j] * grad_val[1];
      }
    }

    // int tid = threadIdx.x + threadIdx.y * blockDim.x;
    // int nthreads = blockDim.x * blockDim.y;
    // __shared__ scalar_t temp_storage[MAX_BLOCK_SIZE];
    // if (n < nthreads) {
    //   int lane = tid % warpSize;
    //   int wid = tid / warpSize;
    //   #pragma unroll
    //   for (int j = 0; j <= 1; ++j) {
    //     d_twiddle_val[j][0] = sum_strided(d_twiddle_val[j][0], temp_storage, n, nthreads, tid);
    //     d_twiddle_val[j][1] = sum_strided(d_twiddle_val[j][1], temp_storage, n, nthreads, tid);
    //   }
    //   int reduction_stride = max(warpSize, n);
    //   int n_block_reductions = div_up(nthreads, reduction_stride);
    //   if ((lane % n_block_reductions == 0) && (wid < n)) {
    //     #pragma unroll
    //     for (int j = 0; j <= 1; ++j) {
    //       atomicAdd(&d_twiddle_expanded_a[j][0][tid / n_block_reductions], d_twiddle_val[j][0]);
    //       atomicAdd(&d_twiddle_expanded_a[j][1][tid / n_block_reductions], d_twiddle_val[j][1]);
    //     }
    //   }
    // } else {
    //   #pragma unroll
    //   for (int j = 0; j <= 1; ++j) {
    //     atomicAdd(&d_twiddle_expanded_a[j][0][i], d_twiddle_val[j][0]);
    //     atomicAdd(&d_twiddle_expanded_a[j][1][i], d_twiddle_val[j][1]);
    //   }
    // }

    // Warp reduction
    for (int offset = warpSize / 2; offset >= n; offset /= 2) {
      #pragma unroll
      for (int j = 0; j <= 1; ++j) {
        d_twiddle_val[j][0] += __shfl_down_sync(FULL_MASK, d_twiddle_val[j][0], offset);
        d_twiddle_val[j][1] += __shfl_down_sync(FULL_MASK, d_twiddle_val[j][1], offset);
      }
    }
    __shared__ scalar_t s_d_twiddle[MAX_BLOCK_SIZE * 4];
    // // const scalar_t (*temp)[n] = (scalar_t (*)[n])(&s_d_twiddle[0]);
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int nthreads = blockDim.x * blockDim.y;
    int lane = tid % warpSize;
    int wid = tid / warpSize;

    if (n < nthreads) {
      __syncthreads();
      s_d_twiddle[tid] = 0;
      s_d_twiddle[tid + MAX_BLOCK_SIZE] = 0;
      s_d_twiddle[tid + 2 * MAX_BLOCK_SIZE] = 0;
      s_d_twiddle[tid + 3 * MAX_BLOCK_SIZE] = 0;
      __syncthreads();
      if (lane < n) {
        atomicAdd(&s_d_twiddle[i], d_twiddle_val[0][0]);
        atomicAdd(&s_d_twiddle[i + MAX_BLOCK_SIZE], d_twiddle_val[0][1]);
        atomicAdd(&s_d_twiddle[i + 2 * MAX_BLOCK_SIZE], d_twiddle_val[1][0]);
        atomicAdd(&s_d_twiddle[i + 3 * MAX_BLOCK_SIZE], d_twiddle_val[1][1]);
      }
      __syncthreads();
      if (tid < n) {
        atomicAdd(&d_twiddle_expanded_a[0][0][i], s_d_twiddle[i]);
        atomicAdd(&d_twiddle_expanded_a[0][1][i], s_d_twiddle[i + MAX_BLOCK_SIZE]);
        atomicAdd(&d_twiddle_expanded_a[1][0][i], s_d_twiddle[i + 2 * MAX_BLOCK_SIZE]);
        atomicAdd(&d_twiddle_expanded_a[1][1][i], s_d_twiddle[i + 3 * MAX_BLOCK_SIZE]);
      }
    } else {
      #pragma unroll
      for (int j = 0; j <= 1; ++j) {
        atomicAdd(&d_twiddle_expanded_a[j][0][i], d_twiddle_val[j][0]);
        atomicAdd(&d_twiddle_expanded_a[j][1][i], d_twiddle_val[j][1]);
      }
    }


    // // Block reduction
    // if (n < nthreads) {
    // // if (n < 0) {
    //   int reduction_stride = max(warpSize, n);
    //   int n_block_reductions = div_up(nthreads, reduction_stride);
    //   if (lane < n) {
    //     // When filling in the shared memory, we assume that n is a power of 2,
    //     // otherwise we might have uninitialized values in the array.
    //     s_d_twiddle[(tid % n) * n_block_reductions + (tid / reduction_stride)] = d_twiddle_val[0][0];
    //     s_d_twiddle[(tid % n) * n_block_reductions + (tid / reduction_stride) + n * n_block_reductions] = d_twiddle_val[0][1];
    //     s_d_twiddle[(tid % n) * n_block_reductions + (tid / reduction_stride) + 2 * n * n_block_reductions] = d_twiddle_val[1][0];
    //     s_d_twiddle[(tid % n) * n_block_reductions + (tid / reduction_stride) + 3 * n * n_block_reductions] = d_twiddle_val[1][1];
    //   }
    //   __syncthreads();
    //   // if (tid == 0) {
    //   //   for (int j = 0; j < 4 * n * n_block_reductions; ++j) {
    //   //     printf("%i: %f\n", j, s_d_twiddle[j]);
    //   //   }
    //   // }
    //   if (wid < n) {
    //     d_twiddle_val[0][0] = s_d_twiddle[tid];
    //     d_twiddle_val[0][1] = s_d_twiddle[tid + n * n_block_reductions];
    //     d_twiddle_val[1][0] = s_d_twiddle[tid + 2 * n * n_block_reductions];
    //     d_twiddle_val[1][1] = s_d_twiddle[tid + 3 * n * n_block_reductions];
    //     for (int offset = n_block_reductions / 2; offset > 0; offset /= 2) {
    //       #pragma unroll
    //       for (int j = 0; j <= 1; ++j) {
    //         d_twiddle_val[j][0] += __shfl_down_sync(FULL_MASK, d_twiddle_val[j][0], offset);
    //         d_twiddle_val[j][1] += __shfl_down_sync(FULL_MASK, d_twiddle_val[j][1], offset);
    //       }
    //     }
    //     if (lane % n_block_reductions == 0) {
    //       #pragma unroll
    //       for (int j = 0; j <= 1; ++j) {
    //         atomicAdd(&d_twiddle_expanded_a[j][0][tid / n_block_reductions], d_twiddle_val[j][0]);
    //         atomicAdd(&d_twiddle_expanded_a[j][1][tid / n_block_reductions], d_twiddle_val[j][1]);
    //       }
    //     }
    //   }
    // // } else {
    // } else if (lane < n) {
    //   #pragma unroll
    //   for (int j = 0; j <= 1; ++j) {
    //     atomicAdd(&d_twiddle_expanded_a[j][0][i], d_twiddle_val[j][0]);
    //     atomicAdd(&d_twiddle_expanded_a[j][1][i], d_twiddle_val[j][1]);
    //   }
    // }

    // if (lane < n) {
    //   #pragma unroll
    //   for (int j = 0; j <= 1; ++j) {
    //     atomicAdd(&d_twiddle_expanded_a[j][0][i], d_twiddle_val[j][0]);
    //     atomicAdd(&d_twiddle_expanded_a[j][1][i], d_twiddle_val[j][1]);
    //   }
    // }
    // #pragma unroll
    // for (int j = 0; j <= 1; ++j) {
    //   d_twiddle_expanded_a[b_start][j][0][i] = d_twiddle_val[j][0];
    //   d_twiddle_expanded_a[b_start][j][1][i] = d_twiddle_val[j][1];
    // }
  }
}

template <typename scalar_t>
__global__ void butterfly_factor_multiply_complex_backward_cuda_kernel(const at::PackedTensorAccessor<scalar_t, 4> grad_a,
                                                                       const at::PackedTensorAccessor<scalar_t, 4> twiddle_a,
                                                                       const at::PackedTensorAccessor<scalar_t, 4> input_a,
                                                                       at::PackedTensorAccessor<scalar_t, 5> d_twiddle_expanded_a,
                                                                       at::PackedTensorAccessor<scalar_t, 4> d_input_a) {
  const auto batch_size = input_a.size(0);
  const auto n = input_a.size(2);
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    const scalar_t twiddle_val[2][2][2] = {{{twiddle_a[0][0][i][0], twiddle_a[0][0][i][1]},
                                            {twiddle_a[0][1][i][0], twiddle_a[0][1][i][1]}},
                                            {{twiddle_a[1][0][i][0], twiddle_a[1][0][i][1]},
                                            {twiddle_a[1][1][i][0], twiddle_a[1][1][i][1]}}};
    for (int64_t b = blockIdx.y * blockDim.y + threadIdx.y; b < batch_size; b += blockDim.y * gridDim.y) {
      const scalar_t input_val[2][2] = {{input_a[b][0][i][0], input_a[b][0][i][1]},
                                        {input_a[b][1][i][0], input_a[b][1][i][1]}};
      const scalar_t grad_val[2][2] = {{grad_a[b][0][i][0], grad_a[b][0][i][1]},
                                       {grad_a[b][1][i][0], grad_a[b][1][i][1]}};
      #pragma unroll
      for (int j = 0; j <= 1; ++j) {
        d_twiddle_expanded_a[b][j][0][i][0] = grad_val[j][0] * input_val[0][0] + grad_val[j][1] * input_val[0][1];
        d_twiddle_expanded_a[b][j][0][i][1] = -grad_val[j][0] * input_val[0][1] + grad_val[j][1] * input_val[0][0];
        d_twiddle_expanded_a[b][j][1][i][0] = grad_val[j][0] * input_val[1][0] + grad_val[j][1] * input_val[1][1];
        d_twiddle_expanded_a[b][j][1][i][1] = -grad_val[j][0] * input_val[1][1] + grad_val[j][1] * input_val[1][0];
        d_input_a[b][j][i][0] = twiddle_val[0][j][0] * grad_val[0][0] + twiddle_val[0][j][1] * grad_val[0][1]
          + twiddle_val[1][j][0] * grad_val[1][0] + twiddle_val[1][j][1] * grad_val[1][1];
        d_input_a[b][j][i][1] = twiddle_val[0][j][0] * grad_val[0][1] - twiddle_val[0][j][1] * grad_val[0][0]
          + twiddle_val[1][j][0] * grad_val[1][1] - twiddle_val[1][j][1] * grad_val[1][0];
      }
    }
  }
}

void butterfly_factor_multiply_backward_cuda(const at::Tensor& grad, const at::Tensor& twiddle, const at::Tensor& input,
                                             at::Tensor& d_twiddle_expanded, at::Tensor& d_input) {
  const auto batch_size = input.size(0);
  const auto n = input.size(2);
  dim3 block;
  block.x = std::min<int64_t>(MAX_BLOCK_SIZE, n);
  block.y = div_up(MAX_BLOCK_SIZE, block.x);
  dim3 grid(div_up(n, block.x), div_up(batch_size, block.y * WORK_PER_THREAD));
  // AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "butterfly_factor_multiply_backward_cuda", [&] {
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "butterfly_factor_multiply_backward_cuda", [&] {
    switch (input.dim()) {
      case 3:  // real
        {
          const auto grad_a = grad.packed_accessor<scalar_t, 3>();
          const auto twiddle_a = twiddle.packed_accessor<scalar_t, 3>();
          const auto input_a = input.packed_accessor<scalar_t, 3>();
          // auto d_twiddle_expanded_a = d_twiddle_expanded.packed_accessor<scalar_t, 4>();
          auto d_twiddle_expanded_a = d_twiddle_expanded.packed_accessor<scalar_t, 3>();
          auto d_input_a = d_input.packed_accessor<scalar_t, 3>();
          butterfly_factor_multiply_backward_cuda_kernel<scalar_t>
            <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(grad_a, twiddle_a, input_a, d_twiddle_expanded_a, d_input_a);
          break;
        }
      case 4:  // complex
        {
          const auto grad_a = grad.packed_accessor<scalar_t, 4>();
          const auto twiddle_a = twiddle.packed_accessor<scalar_t, 4>();
          const auto input_a = input.packed_accessor<scalar_t, 4>();
          auto d_twiddle_expanded_a = d_twiddle_expanded.packed_accessor<scalar_t, 5>();
          auto d_input_a = d_input.packed_accessor<scalar_t, 4>();
          butterfly_factor_multiply_complex_backward_cuda_kernel<scalar_t>
            <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(grad_a, twiddle_a, input_a, d_twiddle_expanded_a, d_input_a);
          break;
        }
      default:
        AT_ERROR("butterfly_factor_multiply requires input dimension 3 or 4");
    }
  });
  AT_CHECK(cudaGetLastError() == cudaSuccess,
     "butterfly_factor_multiply_backward_cuda failed with error code ",
     cudaGetLastError());
}

template <typename scalar_t>
__global__ void butterfly_multiply_inplace_cuda_kernel(const at::PackedTensorAccessor<scalar_t, 3> twiddle_a,
                                                       at::PackedTensorAccessor<scalar_t, 2> input_a,
                                                       int max_stride) {
  const int batch_size = input_a.size(0);
  const int input_base_idx = blockIdx.x * blockDim.x * 2;
  __shared__ scalar_t s_input[ELEMENTARY_SIZE * 2];
  __shared__ scalar_t s_twiddle[ELEMENTARY_SIZE][2][2];
  int64_t b = blockIdx.y * blockDim.y + threadIdx.y;
  if (b < batch_size) {  // Currently we assume 1 batch per thread block, so all threads in the block should enter (otherwise deadlock)
    for (int i = threadIdx.x; i < max_stride * 2; i += blockDim.x) {
      s_input[i] = input_a[b][input_base_idx + i];
    }
    int i = threadIdx.x;
    for (int stride = 1; stride <= max_stride; stride *= 2) {
      int twiddle_start_idx = stride - 1;
      if (i < stride) {
        s_twiddle[i][0][0] = twiddle_a[twiddle_start_idx + i][0][0];
        s_twiddle[i][0][1] = twiddle_a[twiddle_start_idx + i][0][1];
        s_twiddle[i][1][0] = twiddle_a[twiddle_start_idx + i][1][0];
        s_twiddle[i][1][1] = twiddle_a[twiddle_start_idx + i][1][1];
      }
      int low_order_bits = i & (stride - 1);  // int low_order_bits = i % stride;
      int twiddle_idx = low_order_bits;
      int pos = 2 * (i - low_order_bits) + low_order_bits;
      __syncthreads();
      const scalar_t twiddle_val[2][2] = {{s_twiddle[twiddle_idx][0][0], s_twiddle[twiddle_idx][0][1]},
                                          {s_twiddle[twiddle_idx][1][0], s_twiddle[twiddle_idx][1][1]}};
      const scalar_t input_val[2] = {s_input[pos], s_input[pos + stride]};
      s_input[pos] = twiddle_val[0][0] * input_val[0] + twiddle_val[0][1] * input_val[1];
      s_input[pos + stride] = twiddle_val[1][0] * input_val[0] + twiddle_val[1][1] * input_val[1];
    }
    __syncthreads();
    for (int i = threadIdx.x; i < max_stride * 2; i += blockDim.x) {
      input_a[b][input_base_idx + i] = s_input[i];
    }
  }
}

template <typename scalar_t>
__global__ void butterfly_multiply_inplace_onestep_cuda_kernel(const at::PackedTensorAccessor<scalar_t, 3> twiddle_a,
                                                               at::PackedTensorAccessor<scalar_t, 2> input_a,
                                                               int stride) {
  const int batch_size = input_a.size(0);
  int twiddle_start_idx = stride - 1;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int low_order_bits = i & (stride - 1);  // int low_order_bits = i % stride;
  int twiddle_idx = twiddle_start_idx + low_order_bits;
  int pos = 2 * (i - low_order_bits) + low_order_bits;
  const scalar_t twiddle_val[2][2] = {{twiddle_a[twiddle_idx][0][0], twiddle_a[twiddle_idx][0][1]},
                                      {twiddle_a[twiddle_idx][1][0], twiddle_a[twiddle_idx][1][1]}};
  for (int64_t b = blockIdx.y * blockDim.y + threadIdx.y; b < batch_size; b += blockDim.y * gridDim.y) {
    const scalar_t input_val[2] = {input_a[b][pos], input_a[b][pos + stride]};
    input_a[b][pos] = twiddle_val[0][0] * input_val[0] + twiddle_val[0][1] * input_val[1];
    input_a[b][pos + stride] = twiddle_val[1][0] * input_val[0] + twiddle_val[1][1] * input_val[1];
  }
}

void butterfly_multiply_inplace_cuda(const at::Tensor& twiddle, at::Tensor& input) {
  const int batch_size = input.size(0);
  const int n = input.size(1);
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "butterfly_multiply_inplace_cuda", [&] {
    switch (input.dim()) {
      case 2:  // real
        {
          const auto twiddle_a = twiddle.packed_accessor<scalar_t, 3>();
          auto input_a = input.packed_accessor<scalar_t, 2>();
          int stride = std::min<int>(ELEMENTARY_SIZE, n / 2);
          dim3 block(stride);
          dim3 grid(div_up(n / 2, stride), batch_size);
          butterfly_multiply_inplace_cuda_kernel<scalar_t>
            <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, input_a, stride);
          for (stride *= 2; stride <= n / 2; stride *= 2) {
            dim3 block(MAX_BLOCK_SIZE / 2);
            dim3 grid(div_up(n / 2, MAX_BLOCK_SIZE / 2), div_up(batch_size, WORK_PER_THREAD));
            butterfly_multiply_inplace_onestep_cuda_kernel<scalar_t>
              <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, input_a, stride);
          }
          break;
        }
      case 3:  // complex
        {
          const auto twiddle_a = twiddle.packed_accessor<scalar_t, 4>();
          auto input_a = input.packed_accessor<scalar_t, 3>();
          AT_ERROR("Not implemented");
          // butterfly_multiply_inplace_complex_cuda_kernel<scalar_t>
          //   <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, input_a, output_a);
          break;
        }
      default:
        AT_ERROR("butterfly_multiply_inplace requires input dimension 2 or 3");
    }
  });
  AT_CHECK(cudaGetLastError() == cudaSuccess,
     "butterfly_multiply_inplace_cuda failed with error code ",
     cudaGetLastError());
}

template <int LENGTH, typename T>
__device__ __forceinline__ void sum_strided_atomic(T (&val)[LENGTH], T *storage, int stride, int nthreads, int tid) {
  // Warp reduction
  for (int offset = warpSize / 2; offset >= stride; offset /= 2) {
    #pragma unroll
    for (int j = 0; j < LENGTH; j++) {
      val[j] += __shfl_down_sync(FULL_MASK, val[j], offset);
    }
  }
  // Block reduction
  __syncthreads();  // Need this, otherwise might overwrite before other threads can read twiddle values
  if (tid < stride) {
    #pragma unroll
    for (int j = 0; j < LENGTH; j++) {
      storage[j * stride + tid] = 0;
    }
  }
  __syncthreads();
  int lane = tid & (warpSize - 1);  // int lane = tid % waprSize;
  if (lane < stride) {
    #pragma unroll
    for (int j = 0; j < LENGTH; j++) {
      // atomicAdd(&storage[j * stride + tid % stride], val[j]);
      atomicAdd(&storage[j * stride + (tid & (stride - 1))], val[j]);
    }
  }
  __syncthreads();
}

/* Sum elements that are @stride apart by exchanging, using shared memory.
   After the function, threads with @tid < n_block_reductions * stride and @tid % n_block_reductions == 0
   contains the sums.
 */
template <int LENGTH, typename T>
__device__ __forceinline__ void sum_strided_exchange(T (&val)[LENGTH], T *storage, int log_stride, int nthreads, int tid) {
  int stride = 1 << log_stride;
  // Warp reduction
  for (int offset = warpSize / 2; offset >= stride; offset /= 2) {
    #pragma unroll
    for (int j = 0; j < LENGTH; j++) {
      val[j] += __shfl_down_sync(FULL_MASK, val[j], offset);
    }
  }
  int block_reduction_stride = max(warpSize, stride);
  // int n_block_reductions = div_up(nthreads, block_reduction_stride);
  int n_block_reductions = (nthreads + block_reduction_stride - 1) >> max(5, log_stride);
  int lane = tid % warpSize;
  __syncthreads();  // Otherwise previous reads might be wrong
  if ((tid < nthreads) && (lane < stride)) {
    #pragma unroll
    for (int j = 0; j < LENGTH; j++) {
      // storage[j * nthreads + (tid % block_reduction_stride) * n_block_reductions + (tid / block_reduction_stride)] = val[j];
      storage[j * nthreads + (tid & (block_reduction_stride - 1)) * n_block_reductions + (tid / block_reduction_stride)] = val[j];
    }
  }
  __syncthreads();
  if (tid < n_block_reductions * stride) {
    #pragma unroll
    for (int j = 0; j < LENGTH; j++) {
      val[j] = storage[j * nthreads + tid];
    }
    for (int offset = n_block_reductions / 2; offset > 0; offset /= 2) {
      #pragma unroll
      for (int j = 0; j < LENGTH; j++) {
        val[j] += __shfl_down_sync(FULL_MASK, val[j], offset);
      }
    }
  }
}

template <typename scalar_t>
__global__ void butterfly_multiply_inplace_backward_cuda_kernel(const at::PackedTensorAccessor<scalar_t, 3> twiddle_a,
                                                                at::PackedTensorAccessor<double, 2> output_a,
                                                                at::PackedTensorAccessor<scalar_t, 3> d_twiddle_a,
                                                                at::PackedTensorAccessor<scalar_t, 2> d_input_a,
                                                                int max_stride) {
  const int batch_size = output_a.size(0);
  const int input_base_idx = blockIdx.x * blockDim.x * 2;
  __shared__ double s_output[ELEMENTARY_SIZE * 2];
  __shared__ scalar_t s_grad[ELEMENTARY_SIZE * 2];
  __shared__ scalar_t s_twiddle[ELEMENTARY_SIZE][2][2];
  __shared__ scalar_t s_d_twiddle[ELEMENTARY_SIZE * 4];
  int64_t b = blockIdx.y * blockDim.y + threadIdx.y;
  if (b < batch_size) {  // Currently we assume 1 batch per thread block, so all threads in the block should enter (otherwise deadlock)
    for (int i = threadIdx.x; i < max_stride * 2; i += blockDim.x) {
      s_output[i] = output_a[b][input_base_idx + i];
      s_grad[i] = d_input_a[b][input_base_idx + i];
    }
    int i = threadIdx.x;
    for (int stride = max_stride; stride > 0; stride /= 2) {
      int twiddle_start_idx = stride - 1;
      if (i < stride) {
        s_twiddle[i][0][0] = twiddle_a[twiddle_start_idx + i][0][0];
        s_twiddle[i][0][1] = twiddle_a[twiddle_start_idx + i][0][1];
        s_twiddle[i][1][0] = twiddle_a[twiddle_start_idx + i][1][0];
        s_twiddle[i][1][1] = twiddle_a[twiddle_start_idx + i][1][1];
      }
      int low_order_bits = i & (stride - 1);  // int low_order_bits = i % stride;
      int twiddle_idx = low_order_bits;
      int pos = 2 * (i - low_order_bits) + low_order_bits;
      __syncthreads();
      const scalar_t twiddle_val[2][2] = {{s_twiddle[twiddle_idx][0][0], s_twiddle[twiddle_idx][0][1]},
                                          {s_twiddle[twiddle_idx][1][0], s_twiddle[twiddle_idx][1][1]}};
      const scalar_t grad_val[2] = {s_grad[pos], s_grad[pos + stride]};
      s_grad[pos] = twiddle_val[0][0] * grad_val[0] + twiddle_val[1][0] * grad_val[1];
      s_grad[pos + stride] = twiddle_val[0][1] * grad_val[0] + twiddle_val[1][1] * grad_val[1];
      const double output_val[2] = {s_output[pos], s_output[pos + stride]};
      const double twiddle_det_inv = 1.0 / ((double)twiddle_val[0][0] * (double)twiddle_val[1][1] - (double)twiddle_val[0][1] * (double)twiddle_val[1][0]);
      const double input_val[2] = {((double)twiddle_val[1][1] * output_val[0] - (double)twiddle_val[0][1] * output_val[1]) * twiddle_det_inv,
                                  (-(double)twiddle_val[1][0] * output_val[0] + (double)twiddle_val[0][0] * output_val[1]) * twiddle_det_inv};
      s_output[pos] = input_val[0];
      s_output[pos + stride] = input_val[1];
      scalar_t d_twiddle_val[2][2] = {{(scalar_t)(grad_val[0] * input_val[0]),
                                       (scalar_t)(grad_val[0] * input_val[1])},
                                      {(scalar_t)(grad_val[1] * input_val[0]),
                                       (scalar_t)(grad_val[1] * input_val[1])}};
      int tid = threadIdx.x + threadIdx.y * blockDim.x;
      int nthreads = blockDim.x * blockDim.y;
      sum_strided_atomic(reinterpret_cast<scalar_t (&)[4]>(d_twiddle_val), (scalar_t *)s_d_twiddle, stride, nthreads, tid);
      if (tid < stride) {
        atomicAdd(&d_twiddle_a[twiddle_start_idx + twiddle_idx][0][0], s_d_twiddle[twiddle_idx]);
        atomicAdd(&d_twiddle_a[twiddle_start_idx + twiddle_idx][0][1], s_d_twiddle[twiddle_idx + stride]);
        atomicAdd(&d_twiddle_a[twiddle_start_idx + twiddle_idx][1][0], s_d_twiddle[twiddle_idx + 2 * stride]);
        atomicAdd(&d_twiddle_a[twiddle_start_idx + twiddle_idx][1][1], s_d_twiddle[twiddle_idx + 3 * stride]);
      }
      __syncthreads();  // Otherwise s_d_twiddle will be overwritten with s_twiddle before some thread can read
    }
    for (int i = threadIdx.x; i < max_stride * 2; i += blockDim.x) {
      d_input_a[b][input_base_idx + i] = s_grad[i];
    }
  }
}

template <typename scalar_t>
__global__ void butterfly_multiply_inplace_backward_onestep_cuda_kernel(const at::PackedTensorAccessor<scalar_t, 3> twiddle_a,
                                                                        at::PackedTensorAccessor<double, 2> output_a,
                                                                        at::PackedTensorAccessor<scalar_t, 3> d_twiddle_a,
                                                                        at::PackedTensorAccessor<scalar_t, 2> d_input_a,
                                                                        int stride) {
  const int batch_size = output_a.size(0);
  const int n = output_a.size(1);
  int twiddle_start_idx = stride - 1;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i > n) return;
  int low_order_bits = i & (stride - 1);  // int low_order_bits = i % stride;
  int twiddle_idx = twiddle_start_idx + low_order_bits;
  int pos = 2 * (i - low_order_bits) + low_order_bits;
  const scalar_t twiddle_val[2][2] = {{twiddle_a[twiddle_idx][0][0], twiddle_a[twiddle_idx][0][1]},
                                      {twiddle_a[twiddle_idx][1][0], twiddle_a[twiddle_idx][1][1]}};
  scalar_t d_twiddle_val[2][2] = {{0, 0}, {0, 0}};
  for (int64_t b = blockIdx.y * blockDim.y + threadIdx.y; b < batch_size; b += blockDim.y * gridDim.y) {
    const scalar_t grad_val[2] = {d_input_a[b][pos], d_input_a[b][pos + stride]};
    d_input_a[b][pos] = twiddle_val[0][0] * grad_val[0] + twiddle_val[1][0] * grad_val[1];
    d_input_a[b][pos + stride] = twiddle_val[0][1] * grad_val[0] + twiddle_val[1][1] * grad_val[1];
    const double output_val[2] = {output_a[b][pos], output_a[b][pos + stride]};
    const double twiddle_det_inv = 1.0 / ((double)twiddle_val[0][0] * (double)twiddle_val[1][1] - (double)twiddle_val[0][1] * (double)twiddle_val[1][0]);
    const double input_val[2] = {((double)twiddle_val[1][1] * output_val[0] - (double)twiddle_val[0][1] * output_val[1]) * twiddle_det_inv,
                                 (-(double)twiddle_val[1][0] * output_val[0] + (double)twiddle_val[0][0] * output_val[1]) * twiddle_det_inv};
    output_a[b][pos] = input_val[0];
    output_a[b][pos + stride] = input_val[1];
    d_twiddle_val[0][0] += grad_val[0] * input_val[0];
    d_twiddle_val[0][1] += grad_val[0] * input_val[1];
    d_twiddle_val[1][0] += grad_val[1] * input_val[0];
    d_twiddle_val[1][1] += grad_val[1] * input_val[1];
  }
  atomicAdd(&d_twiddle_a[twiddle_idx][0][0], d_twiddle_val[0][0]);
  atomicAdd(&d_twiddle_a[twiddle_idx][0][1], d_twiddle_val[0][1]);
  atomicAdd(&d_twiddle_a[twiddle_idx][1][0], d_twiddle_val[1][0]);
  atomicAdd(&d_twiddle_a[twiddle_idx][1][1], d_twiddle_val[1][1]);
}

void butterfly_multiply_inplace_backward_cuda(const at::Tensor& grad, const at::Tensor& twiddle, at::Tensor& output,
                                              at::Tensor& d_twiddle, at::Tensor& d_input) {
  const int batch_size = output.size(0);
  const int n = output.size(1);
  AT_DISPATCH_FLOATING_TYPES(grad.scalar_type(), "butterfly_multiply_inplace_backward_cuda", [&] {
    switch (grad.dim()) {
      case 2:  // real
        {
          const auto twiddle_a = twiddle.packed_accessor<scalar_t, 3>();
          auto output_a = output.packed_accessor<double, 2>();
          auto d_twiddle_a = d_twiddle.packed_accessor<scalar_t, 3>();
          auto d_input_a = d_input.packed_accessor<scalar_t, 2>();
          int stride = n/2;
          for (; stride > ELEMENTARY_SIZE; stride /= 2) {
          // for (; stride > 0; stride /= 2) {
            dim3 block(MAX_BLOCK_SIZE / 2);
            dim3 grid(div_up(n / 2, MAX_BLOCK_SIZE / 2), div_up(batch_size, WORK_PER_THREAD));
            butterfly_multiply_inplace_backward_onestep_cuda_kernel<scalar_t>
              <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, output_a, d_twiddle_a, d_input_a, stride);
          }
          dim3 block(stride);
          dim3 grid(div_up(n / 2, stride), batch_size);
          butterfly_multiply_inplace_backward_cuda_kernel<scalar_t>
            <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, output_a, d_twiddle_a, d_input_a, stride);
          break;
        }
      case 3:  // complex
        {
          const auto twiddle_a = twiddle.packed_accessor<scalar_t, 4>();
          auto output_a = output.packed_accessor<scalar_t, 3>();
          AT_ERROR("Not implemented");
          // butterfly_multiply_inplace_backward_complex_cuda_kernel<scalar_t>
          //   <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, output_a, output_a);
          break;
        }
      default:
        AT_ERROR("butterfly_multiply_inplace_backward requires input dimension 2 or 3");
    }
  });
  AT_CHECK(cudaGetLastError() == cudaSuccess,
     "butterfly_multiply_inplace_backward_cuda failed with error code ",
     cudaGetLastError());
}

template <typename scalar_t, bool increasing_stride, bool return_intermediates>
__global__ void butterfly_multiply_intermediate_cuda_kernel(const at::PackedTensorAccessor<scalar_t, 4> twiddle_a,
                                                            at::PackedTensorAccessor<scalar_t, 4> output_a,
                                                            int log_max_stride,
                                                            int log_n) {
  const int batch_size = output_a.size(1);
  const int s = blockIdx.z;
  const int max_stride = 1 << log_max_stride;
  const int input_base_idx = blockIdx.x * blockDim.x * 2;
  __shared__ scalar_t s_input[ELEMENTARY_SIZE * 2];
  __shared__ scalar_t s_twiddle[ELEMENTARY_SIZE][2][2];
  int b = blockIdx.y * blockDim.y + threadIdx.y;
  if (b < batch_size) {  // Currently we assume 1 batch per thread block, so all threads in the block should enter (otherwise deadlock)
    int first_idx = increasing_stride ? 0 : log_n - 1 - log_max_stride;
    for (int i = threadIdx.x; i < max_stride * 2; i += blockDim.x) {
      s_input[i] = output_a[first_idx][b][s][input_base_idx + i];
    }
    int i = threadIdx.x;
    for (int idx = first_idx; idx <= first_idx + log_max_stride; ++idx) {
      int log_stride = increasing_stride ? idx : log_n - 1 - idx;
      int stride = 1 << log_stride;
      int twiddle_start_idx = stride - 1;
      if (i < stride) {
        s_twiddle[i][0][0] = twiddle_a[s][twiddle_start_idx + i][0][0];
        s_twiddle[i][0][1] = twiddle_a[s][twiddle_start_idx + i][0][1];
        s_twiddle[i][1][0] = twiddle_a[s][twiddle_start_idx + i][1][0];
        s_twiddle[i][1][1] = twiddle_a[s][twiddle_start_idx + i][1][1];
      }
      int low_order_bits = i & (stride - 1);  // int low_order_bits = i % stride;
      int twiddle_idx = low_order_bits;
      int pos = 2 * (i - low_order_bits) + low_order_bits;
      __syncthreads();
      const scalar_t twiddle_val[2][2] = {{s_twiddle[twiddle_idx][0][0], s_twiddle[twiddle_idx][0][1]},
                                          {s_twiddle[twiddle_idx][1][0], s_twiddle[twiddle_idx][1][1]}};
      __syncthreads();  // otherwise some thread might go back to writing to s_twiddle before other thread can read
      const scalar_t input_val[2] = {s_input[pos], s_input[pos + stride]};
      thrust::tie(s_input[pos], s_input[pos + stride]) = mult2x2(twiddle_val[0][0], twiddle_val[0][1], twiddle_val[1][0], twiddle_val[1][1],
                                                                 input_val[0], input_val[1]);
      if (return_intermediates || idx == first_idx + log_max_stride) {
        output_a[idx+1][b][s][input_base_idx + pos] = s_input[pos];
        output_a[idx+1][b][s][input_base_idx + pos + stride] = s_input[pos + stride];
      }
    }
  }
}

template <typename scalar_t, bool increasing_stride, bool return_intermediates>
__global__ void butterfly_multiply_intermediate_complex_cuda_kernel(const at::PackedTensorAccessor<scalar_t, 5> twiddle_a,
                                                                    at::PackedTensorAccessor<scalar_t, 5> output_a,
                                                                    int log_max_stride,
                                                                    int log_n) {
  using complex_t = thrust::complex<scalar_t>;
  const int batch_size = output_a.size(1);
  const int s = blockIdx.z;
  const int max_stride = 1 << log_max_stride;
  const int input_base_idx = blockIdx.x * blockDim.x * 2;
  // __shared__ complex_t s_input[ELEMENTARY_SIZE * 2];
  __shared__ scalar_t s_input_storage[ELEMENTARY_SIZE * 2][2];
  complex_t* s_input = (complex_t *)&s_input_storage[0];  // To avoid warning about race-condition when initializing complex_t
  // __shared__ complex_t s_twiddle[ELEMENTARY_SIZE][2][2];
  __shared__ scalar_t s_twiddle_storage[ELEMENTARY_SIZE][2][2][2];
  complex_t (* s_twiddle)[2][2] = (complex_t (*)[2][2])&s_twiddle_storage[0];  // To avoid warning about race-condition when initializing complex_t
  int b = blockIdx.y * blockDim.y + threadIdx.y;
  if (b < batch_size) {  // Currently we assume 1 batch per thread block, so all threads in the block should enter (otherwise deadlock)
    int first_idx = increasing_stride ? 0 : log_n - 1 - log_max_stride;
    for (int i = threadIdx.x; i < max_stride * 2; i += blockDim.x) {
      s_input[i] = complex_t(output_a[first_idx][b][s][input_base_idx + i][0], output_a[first_idx][b][s][input_base_idx + i][1]);
    }
    int i = threadIdx.x;
    for (int idx = first_idx; idx <= first_idx + log_max_stride; ++idx) {
      int log_stride = increasing_stride ? idx : log_n - 1 - idx;
      int stride = 1 << log_stride;
      int twiddle_start_idx = stride - 1;
      if (i < stride) {
        s_twiddle[i][0][0] = complex_t(twiddle_a[s][twiddle_start_idx + i][0][0][0], twiddle_a[s][twiddle_start_idx + i][0][0][1]);
        s_twiddle[i][0][1] = complex_t(twiddle_a[s][twiddle_start_idx + i][0][1][0], twiddle_a[s][twiddle_start_idx + i][0][1][1]);
        s_twiddle[i][1][0] = complex_t(twiddle_a[s][twiddle_start_idx + i][1][0][0], twiddle_a[s][twiddle_start_idx + i][1][0][1]);
        s_twiddle[i][1][1] = complex_t(twiddle_a[s][twiddle_start_idx + i][1][1][0], twiddle_a[s][twiddle_start_idx + i][1][1][1]);
      }
      int low_order_bits = i & (stride - 1);  // int low_order_bits = i % stride;
      int twiddle_idx = low_order_bits;
      int pos = 2 * (i - low_order_bits) + low_order_bits;
      __syncthreads();
      const complex_t twiddle_val[2][2] = {{s_twiddle[twiddle_idx][0][0], s_twiddle[twiddle_idx][0][1]},
                                           {s_twiddle[twiddle_idx][1][0], s_twiddle[twiddle_idx][1][1]}};
      __syncthreads();  // otherwise some thread might go back to writing to s_twiddle before other thread can read
      const complex_t input_val[2] = {s_input[pos], s_input[pos + stride]};
      thrust::tie(s_input[pos], s_input[pos + stride]) = mult2x2(twiddle_val[0][0], twiddle_val[0][1], twiddle_val[1][0], twiddle_val[1][1],
                                                                 input_val[0], input_val[1]);
      if (return_intermediates || idx == first_idx + log_max_stride) {
        output_a[idx+1][b][s][input_base_idx + pos][0] = s_input[pos].real();
        output_a[idx+1][b][s][input_base_idx + pos][1] = s_input[pos].imag();
        output_a[idx+1][b][s][input_base_idx + pos + stride][0] = s_input[pos + stride].real();
        output_a[idx+1][b][s][input_base_idx + pos + stride][1] = s_input[pos + stride].imag();
      }
    }
  }
}

template <typename scalar_t, bool increasing_stride>
__global__ void butterfly_multiply_intermediate_onestep_cuda_kernel(const at::PackedTensorAccessor<scalar_t, 4> twiddle_a,
                                                                    at::PackedTensorAccessor<scalar_t, 4> output_a,
                                                                    int log_stride,
                                                                    int log_n) {
  const int batch_size = output_a.size(1);
  const int s = blockIdx.z;
  const int idx = increasing_stride ? log_stride : (log_n - 1 - log_stride);  // Index to access output_a
  const int stride = 1 << log_stride;
  int twiddle_start_idx = stride - 1;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int low_order_bits = i & (stride - 1);  // int low_order_bits = i % stride;
  int twiddle_idx = twiddle_start_idx + low_order_bits;
  int pos = 2 * (i - low_order_bits) + low_order_bits;
  const scalar_t twiddle_val[2][2] = {{twiddle_a[s][twiddle_idx][0][0], twiddle_a[s][twiddle_idx][0][1]},
                                      {twiddle_a[s][twiddle_idx][1][0], twiddle_a[s][twiddle_idx][1][1]}};
  for (int b = blockIdx.y * blockDim.y + threadIdx.y; b < batch_size; b += blockDim.y * gridDim.y) {
    const scalar_t input_val[2] = {output_a[idx][b][s][pos], output_a[idx][b][s][pos + stride]};
    output_a[idx+1][b][s][pos] = twiddle_val[0][0] * input_val[0] + twiddle_val[0][1] * input_val[1];
    output_a[idx+1][b][s][pos + stride] = twiddle_val[1][0] * input_val[0] + twiddle_val[1][1] * input_val[1];
  }
}

template <typename scalar_t, bool increasing_stride>
__global__ void butterfly_multiply_intermediate_onestep_complex_cuda_kernel(const at::PackedTensorAccessor<scalar_t, 5> twiddle_a,
                                                                            at::PackedTensorAccessor<scalar_t, 5> output_a,
                                                                            int log_stride,
                                                                            int log_n) {
  using complex_t = thrust::complex<scalar_t>;
  const int batch_size = output_a.size(1);
  const int s = blockIdx.z;
  const int idx = increasing_stride ? log_stride : (log_n - 1 - log_stride);  // Index to access output_a
  const int stride = 1 << log_stride;
  int twiddle_start_idx = stride - 1;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int low_order_bits = i & (stride - 1);  // int low_order_bits = i % stride;
  int twiddle_idx = twiddle_start_idx + low_order_bits;
  int pos = 2 * (i - low_order_bits) + low_order_bits;
  const complex_t twiddle_val[2][2] =
    {{complex_t(twiddle_a[s][twiddle_idx][0][0][0], twiddle_a[s][twiddle_idx][0][0][1]),
      complex_t(twiddle_a[s][twiddle_idx][0][1][0], twiddle_a[s][twiddle_idx][0][1][1])},
     {complex_t(twiddle_a[s][twiddle_idx][1][0][0], twiddle_a[s][twiddle_idx][1][0][1]),
      complex_t(twiddle_a[s][twiddle_idx][1][1][0], twiddle_a[s][twiddle_idx][1][1][1])}};
  for (int b = blockIdx.y * blockDim.y + threadIdx.y; b < batch_size; b += blockDim.y * gridDim.y) {
    const complex_t input_val[2] =
      {complex_t(output_a[idx][b][s][pos][0], output_a[idx][b][s][pos][1]),
       complex_t(output_a[idx][b][s][pos + stride][0], output_a[idx][b][s][pos + stride][1])};
    const complex_t output_val[2] =
      {twiddle_val[0][0] * input_val[0] + twiddle_val[0][1] * input_val[1],
       twiddle_val[1][0] * input_val[0] + twiddle_val[1][1] * input_val[1]};
    output_a[idx+1][b][s][pos][0] = output_val[0].real();
    output_a[idx+1][b][s][pos][1] = output_val[0].imag();
    output_a[idx+1][b][s][pos + stride][0] = output_val[1].real();
    output_a[idx+1][b][s][pos + stride][1] = output_val[1].imag();
  }
}

void butterfly_multiply_intermediate_cuda(const at::Tensor& twiddle, at::Tensor& output, bool increasing_stride, bool return_intermediates) {
  const int batch_size = output.size(1);
  const int nstack = twiddle.size(0);
  const int n = output.size(3);
  const int log_n = int(log2((double) n));
  const bool complex = output.dim() == 5;
  AT_DISPATCH_FLOATING_TYPES(output.scalar_type(), "butterfly_multiply_intermediate_cuda", [&] {
    if (!complex) {  // real
      const auto twiddle_a = twiddle.packed_accessor<scalar_t, 4>();
      auto output_a = output.packed_accessor<scalar_t, 4>();
      if (increasing_stride) {
        int stride = std::min<int>(ELEMENTARY_SIZE, n / 2);
        int log_stride = int(log2((double) stride));
        dim3 block(stride);
        dim3 grid(div_up(n / 2, stride), batch_size, nstack);
        return_intermediates ? butterfly_multiply_intermediate_cuda_kernel<scalar_t, true, true>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, output_a, log_stride, log_n)
                             : butterfly_multiply_intermediate_cuda_kernel<scalar_t, true, false>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, output_a, log_stride, log_n);
        for (log_stride++; log_stride <= log_n - 1; ++log_stride) {
          dim3 block(MAX_BLOCK_SIZE / 2);
          dim3 grid(div_up(n / 2, MAX_BLOCK_SIZE / 2), div_up(batch_size, WORK_PER_THREAD), nstack);
          butterfly_multiply_intermediate_onestep_cuda_kernel<scalar_t, true>
            <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, output_a, log_stride, log_n);
        }
      } else {
        int log_stride = log_n - 1;
        for (; (1 << log_stride) > ELEMENTARY_SIZE; --log_stride) {
          dim3 block(MAX_BLOCK_SIZE / 2);
          dim3 grid(div_up(n / 2, MAX_BLOCK_SIZE / 2), div_up(batch_size, WORK_PER_THREAD), nstack);
          butterfly_multiply_intermediate_onestep_cuda_kernel<scalar_t, false>
            <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, output_a, log_stride, log_n);
        }
        int stride = 1 << log_stride;
        dim3 block(stride);
        dim3 grid(div_up(n / 2, stride), batch_size, nstack);
        return_intermediates ? butterfly_multiply_intermediate_cuda_kernel<scalar_t, false, true>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, output_a, log_stride, log_n)
                             : butterfly_multiply_intermediate_cuda_kernel<scalar_t, false, false>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, output_a, log_stride, log_n);
      }
    } else {  // complex
      const auto twiddle_a = twiddle.packed_accessor<scalar_t, 5>();
      auto output_a = output.packed_accessor<scalar_t, 5>();
      if (increasing_stride) {
        int stride = std::min<int>(ELEMENTARY_SIZE, n / 2);
        int log_stride = int(log2((double) stride));
        dim3 block(stride);
        dim3 grid(div_up(n / 2, stride), batch_size, nstack);
        return_intermediates ? butterfly_multiply_intermediate_complex_cuda_kernel<scalar_t, true, true>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, output_a, log_stride, log_n)
                             : butterfly_multiply_intermediate_complex_cuda_kernel<scalar_t, true, false>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, output_a, log_stride, log_n);
        for (log_stride++; log_stride <= log_n - 1; ++log_stride) {
          dim3 block(MAX_BLOCK_SIZE / 2);
          dim3 grid(div_up(n / 2, MAX_BLOCK_SIZE / 2), div_up(batch_size, WORK_PER_THREAD), nstack);
          butterfly_multiply_intermediate_onestep_complex_cuda_kernel<scalar_t, true>
            <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, output_a, log_stride, log_n);
        }
      } else {
        int log_stride = log_n - 1;
        for (; (1 << log_stride) > ELEMENTARY_SIZE; --log_stride) {
          dim3 block(MAX_BLOCK_SIZE / 2);
          dim3 grid(div_up(n / 2, MAX_BLOCK_SIZE / 2), div_up(batch_size, WORK_PER_THREAD), nstack);
          butterfly_multiply_intermediate_onestep_complex_cuda_kernel<scalar_t, false>
            <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, output_a, log_stride, log_n);
        }
        int stride = 1 << log_stride;
        dim3 block(stride);
        dim3 grid(div_up(n / 2, stride), batch_size, nstack);
        return_intermediates ? butterfly_multiply_intermediate_complex_cuda_kernel<scalar_t, false, true>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, output_a, log_stride, log_n)
                             : butterfly_multiply_intermediate_complex_cuda_kernel<scalar_t, false, false>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, output_a, log_stride, log_n);
      }
    }
  });
  AT_CHECK(cudaGetLastError() == cudaSuccess,
     "butterfly_multiply_intermediate_cuda failed with error code ",
     cudaGetLastError());
}

template <typename scalar_t, typename accscalar_t, bool increasing_stride>
__global__ void butterfly_multiply_intermediate_backward_cuda_kernel(const at::PackedTensorAccessor<scalar_t, 4> twiddle_a,
                                                                     const at::PackedTensorAccessor<scalar_t, 4> output_a,
                                                                     at::PackedTensorAccessor<scalar_t, 4> d_twiddle_a,
                                                                     at::PackedTensorAccessor<scalar_t, 3> d_input_a,
                                                                     int log_max_stride,
                                                                     int log_n) {
  const int batch_size = output_a.size(1);
  const int s = blockIdx.z;
  const int max_stride = 1 << log_max_stride;
  const int input_base_idx = blockIdx.x * blockDim.x * 2;
  __shared__ scalar_t s_grad[ELEMENTARY_SIZE * 2];
  __shared__ accscalar_t s_twiddle[ELEMENTARY_SIZE][2][2];  // Use accscalar_t instead of scalar_t since we'll reuse the storage for s_d_twiddle
  // __shared__ scalar_t s_d_twiddle[ELEMENTARY_SIZE * 4];
  // accscalar_t (* s_d_twiddle)[2][2] = (accscalar_t (*)[2][2])&s_twiddle[0][0][0];  // Reusing the same storage as s_twiddle, have to be careful if we change the implemetnation.
  accscalar_t* s_d_twiddle = (accscalar_t *)&s_twiddle[0][0][0];  // Reusing the same storage as s_twiddle, have to be careful if we change the implemetnation.
  int b = blockIdx.y * blockDim.y + threadIdx.y;
  if (b < batch_size) {  // Currently we assume 1 batch per thread block, so all threads in the block should enter (otherwise deadlock)
    for (int i = threadIdx.x; i < max_stride * 2; i += blockDim.x) {
      s_grad[i] = d_input_a[b][s][input_base_idx + i];
    }
    int i = threadIdx.x;
    int first_idx = increasing_stride ? 0 : log_n - 1 - log_max_stride;
    for (int idx = first_idx + log_max_stride; idx >= first_idx; --idx) {
      int log_stride = increasing_stride ? idx : log_n - 1 - idx;
      int stride = 1 << log_stride;
      int twiddle_start_idx = stride - 1;
      if (i < stride) {
        s_twiddle[i][0][0] = twiddle_a[s][twiddle_start_idx + i][0][0];
        s_twiddle[i][0][1] = twiddle_a[s][twiddle_start_idx + i][0][1];
        s_twiddle[i][1][0] = twiddle_a[s][twiddle_start_idx + i][1][0];
        s_twiddle[i][1][1] = twiddle_a[s][twiddle_start_idx + i][1][1];
      }
      int low_order_bits = i & (stride - 1);  // int low_order_bits = i % stride;
      int twiddle_idx = low_order_bits;
      int pos = 2 * (i - low_order_bits) + low_order_bits;
      __syncthreads();
      const scalar_t twiddle_val[2][2] = {{s_twiddle[twiddle_idx][0][0], s_twiddle[twiddle_idx][0][1]},
                                          {s_twiddle[twiddle_idx][1][0], s_twiddle[twiddle_idx][1][1]}};
      // Don't need to sync here since we sync later at sum_strided_atomic, so no writing to s_twiddle can occur until then
      const scalar_t grad_val[2] = {s_grad[pos], s_grad[pos + stride]};
      s_grad[pos] = twiddle_val[0][0] * grad_val[0] + twiddle_val[1][0] * grad_val[1];
      s_grad[pos + stride] = twiddle_val[0][1] * grad_val[0] + twiddle_val[1][1] * grad_val[1];
      const scalar_t input_val[2] = {output_a[idx][b][s][input_base_idx + pos], output_a[idx][b][s][input_base_idx + pos + stride]};
      accscalar_t d_twiddle_val[2][2] = {{grad_val[0] * input_val[0], grad_val[0] * input_val[1]},
                                         {grad_val[1] * input_val[0], grad_val[1] * input_val[1]}};
      int tid = threadIdx.x + threadIdx.y * blockDim.x;
      int nthreads = blockDim.x * blockDim.y;
      sum_strided_atomic(reinterpret_cast<accscalar_t (&)[4]>(d_twiddle_val), s_d_twiddle, stride, nthreads, tid);
      if (tid < stride) {
        atomicAdd(&d_twiddle_a[s][twiddle_start_idx + twiddle_idx][0][0], s_d_twiddle[twiddle_idx]);
        atomicAdd(&d_twiddle_a[s][twiddle_start_idx + twiddle_idx][0][1], s_d_twiddle[twiddle_idx + stride]);
        atomicAdd(&d_twiddle_a[s][twiddle_start_idx + twiddle_idx][1][0], s_d_twiddle[twiddle_idx + 2 * stride]);
        atomicAdd(&d_twiddle_a[s][twiddle_start_idx + twiddle_idx][1][1], s_d_twiddle[twiddle_idx + 3 * stride]);
      }
      __syncthreads();  // Otherwise s_d_twiddle will be overwritten with s_twiddle before some thread can read
      // sum_strided_exchange(reinterpret_cast<accscalar_t (&)[4]>(d_twiddle_val), s_d_twiddle, log_stride, nthreads, tid);
      // int block_reduction_stride = max(warpSize, stride);
      // // int n_block_reductions = div_up(nthreads, block_reduction_stride);
      // int n_block_reductions = (nthreads + block_reduction_stride - 1) >> max(5, log_stride);
      // // if ((tid < n_block_reductions * stride) && (tid % n_block_reductions == 0)) {
      // if ((tid < n_block_reductions * stride) && ((tid & (n_block_reductions - 1)) == 0)) {
      //   // atomicAdd(&d_twiddle_a[s][twiddle_start_idx + tid / n_block_reductions][0][0], d_twiddle_val[0][0]);
      //   // Trying to avoid integer division
      //   int log_n_block_reductions = log_max_stride - max(5, log_stride);  // Use the fact that nthreads == max_stride and warpSize == 32
      //   atomicAdd(&d_twiddle_a[s][twiddle_start_idx + (tid >> log_n_block_reductions)][0][0], d_twiddle_val[0][0]);
      //   atomicAdd(&d_twiddle_a[s][twiddle_start_idx + (tid >> log_n_block_reductions)][0][1], d_twiddle_val[0][1]);
      //   atomicAdd(&d_twiddle_a[s][twiddle_start_idx + (tid >> log_n_block_reductions)][1][0], d_twiddle_val[1][0]);
      //   atomicAdd(&d_twiddle_a[s][twiddle_start_idx + (tid >> log_n_block_reductions)][1][1], d_twiddle_val[1][1]);
      // }
    }
    for (int i = threadIdx.x; i < max_stride * 2; i += blockDim.x) {
      d_input_a[b][s][input_base_idx + i] = s_grad[i];
    }
  }
}

template <typename scalar_t, typename accscalar_t, bool increasing_stride>
__global__ void butterfly_multiply_intermediate_backward_complex_cuda_kernel(const at::PackedTensorAccessor<scalar_t, 5> twiddle_a,
                                                                             const at::PackedTensorAccessor<scalar_t, 5> output_a,
                                                                             at::PackedTensorAccessor<scalar_t, 5> d_twiddle_a,
                                                                             at::PackedTensorAccessor<scalar_t, 4> d_input_a,
                                                                             int log_max_stride,
                                                                             int log_n) {
  using complex_t = thrust::complex<scalar_t>;
  using acccomplex_t = thrust::complex<accscalar_t>;
  const int batch_size = output_a.size(1);
  const int s = blockIdx.z;
  const int max_stride = 1 << log_max_stride;
  const int input_base_idx = blockIdx.x * blockDim.x * 2;
  // __shared__ scalar_t s_grad[ELEMENTARY_SIZE * 2][2];
  __shared__ scalar_t s_grad_storage[ELEMENTARY_SIZE * 2][2];
  complex_t* s_grad = (complex_t *)&s_grad_storage[0];  // To avoid warning about race-condition when initializing complex_t
  // __shared__ accscalar_t s_twiddle[ELEMENTARY_SIZE][2][2][2];  // Use accscalar_t instead of scalar_t since we'll reuse the storage for s_d_twiddle
  __shared__ accscalar_t s_twiddle_storage[ELEMENTARY_SIZE][2][2][2];
  acccomplex_t (* s_twiddle)[2][2] = (acccomplex_t (*)[2][2])&s_twiddle_storage[0];  // To avoid warning about race-condition when initializing complex_t
  // __shared__ scalar_t s_d_twiddle[ELEMENTARY_SIZE * 4];
  // acccomplex_t (* s_d_twiddle)[2][2] = (acccomplex_t (*)[2][2])&s_twiddle[0][0][0];  // Reusing the same storage as s_twiddle, have to be careful if we change the implemetnation.
  acccomplex_t* s_d_twiddle = (acccomplex_t *)&s_twiddle[0][0][0];  // Reusing the same storage as s_twiddle, have to be careful if we change the implemetnation.
  int b = blockIdx.y * blockDim.y + threadIdx.y;
  if (b < batch_size) {  // Currently we assume 1 batch per thread block, so all threads in the block should enter (otherwise deadlock)
    for (int i = threadIdx.x; i < max_stride * 2; i += blockDim.x) {
      s_grad[i] = complex_t(d_input_a[b][s][input_base_idx + i][0], d_input_a[b][s][input_base_idx + i][1]);
    }
    int i = threadIdx.x;
    int first_idx = increasing_stride ? 0 : log_n - 1 - log_max_stride;
    for (int idx = first_idx + log_max_stride; idx >= first_idx; --idx) {
      int log_stride = increasing_stride ? idx : log_n - 1 - idx;
      int stride = 1 << log_stride;
      int twiddle_start_idx = stride - 1;
      if (i < stride) {
        s_twiddle[i][0][0] = complex_t(twiddle_a[s][twiddle_start_idx + i][0][0][0], twiddle_a[s][twiddle_start_idx + i][0][0][1]);
        s_twiddle[i][0][1] = complex_t(twiddle_a[s][twiddle_start_idx + i][0][1][0], twiddle_a[s][twiddle_start_idx + i][0][1][1]);
        s_twiddle[i][1][0] = complex_t(twiddle_a[s][twiddle_start_idx + i][1][0][0], twiddle_a[s][twiddle_start_idx + i][1][0][1]);
        s_twiddle[i][1][1] = complex_t(twiddle_a[s][twiddle_start_idx + i][1][1][0], twiddle_a[s][twiddle_start_idx + i][1][1][1]);
      }
      int low_order_bits = i & (stride - 1);  // int low_order_bits = i % stride;
      int twiddle_idx = low_order_bits;
      int pos = 2 * (i - low_order_bits) + low_order_bits;
      __syncthreads();
      const complex_t twiddle_val[2][2] = {{s_twiddle[twiddle_idx][0][0], s_twiddle[twiddle_idx][0][1]},
                                           {s_twiddle[twiddle_idx][1][0], s_twiddle[twiddle_idx][1][1]}};
      // Don't need to sync here since we sync later at sum_strided_atomic, so no writing to s_twiddle can occur until then
      const complex_t grad_val[2] = {s_grad[pos], s_grad[pos + stride]};
      s_grad[pos] = thrust::conj(twiddle_val[0][0]) * grad_val[0] + thrust::conj(twiddle_val[1][0]) * grad_val[1];
      s_grad[pos + stride] = thrust::conj(twiddle_val[0][1]) * grad_val[0] + thrust::conj(twiddle_val[1][1]) * grad_val[1];
      const complex_t input_val[2] =
        {complex_t(output_a[idx][b][s][input_base_idx + pos][0], output_a[idx][b][s][input_base_idx + pos][1]),
         complex_t(output_a[idx][b][s][input_base_idx + pos + stride][0], output_a[idx][b][s][input_base_idx + pos + stride][1])};
      acccomplex_t d_twiddle_val[2][2] =
        {{grad_val[0] * thrust::conj(input_val[0]), grad_val[0] * thrust::conj(input_val[1])},
         {grad_val[1] * thrust::conj(input_val[0]), grad_val[1] * thrust::conj(input_val[1])}};
      int tid = threadIdx.x + threadIdx.y * blockDim.x;
      int nthreads = blockDim.x * blockDim.y;
      sum_strided_atomic(reinterpret_cast<acccomplex_t (&)[4]>(d_twiddle_val), s_d_twiddle, stride, nthreads, tid);
      if (tid < stride) {
        atomicAdd(&d_twiddle_a[s][twiddle_start_idx + twiddle_idx][0][0][0], s_d_twiddle[twiddle_idx].real());
        atomicAdd(&d_twiddle_a[s][twiddle_start_idx + twiddle_idx][0][0][1], s_d_twiddle[twiddle_idx].imag());
        atomicAdd(&d_twiddle_a[s][twiddle_start_idx + twiddle_idx][0][1][0], s_d_twiddle[twiddle_idx + stride].real());
        atomicAdd(&d_twiddle_a[s][twiddle_start_idx + twiddle_idx][0][1][1], s_d_twiddle[twiddle_idx + stride].imag());
        atomicAdd(&d_twiddle_a[s][twiddle_start_idx + twiddle_idx][1][0][0], s_d_twiddle[twiddle_idx + 2 * stride].real());
        atomicAdd(&d_twiddle_a[s][twiddle_start_idx + twiddle_idx][1][0][1], s_d_twiddle[twiddle_idx + 2 * stride].imag());
        atomicAdd(&d_twiddle_a[s][twiddle_start_idx + twiddle_idx][1][1][0], s_d_twiddle[twiddle_idx + 3 * stride].real());
        atomicAdd(&d_twiddle_a[s][twiddle_start_idx + twiddle_idx][1][1][1], s_d_twiddle[twiddle_idx + 3 * stride].imag());
      }
      __syncthreads();  // Otherwise s_d_twiddle will be overwritten with s_twiddle before some thread can read
    }
    for (int i = threadIdx.x; i < max_stride * 2; i += blockDim.x) {
      d_input_a[b][s][input_base_idx + i][0] = s_grad[i].real();
      d_input_a[b][s][input_base_idx + i][1] = s_grad[i].imag();
    }
  }
}

template <typename scalar_t, typename accscalar_t, bool increasing_stride>
__global__ void butterfly_multiply_intermediate_backward_onestep_cuda_kernel(const at::PackedTensorAccessor<scalar_t, 4> twiddle_a,
                                                                             const at::PackedTensorAccessor<scalar_t, 4> output_a,
                                                                             at::PackedTensorAccessor<scalar_t, 4> d_twiddle_a,
                                                                             at::PackedTensorAccessor<scalar_t, 3> d_input_a,
                                                                             int log_stride,
                                                                             int log_n) {
  const int batch_size = output_a.size(1);
  const int s = blockIdx.z;
  const int idx = increasing_stride ? log_stride : (log_n - 1 - log_stride);  // Index to access output_a
  const int n = output_a.size(3);
  int stride = 1 << log_stride;
  int twiddle_start_idx = stride - 1;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i > n) return;
  int low_order_bits = i & (stride - 1);  // int low_order_bits = i % stride;
  int twiddle_idx = twiddle_start_idx + low_order_bits;
  int pos = 2 * (i - low_order_bits) + low_order_bits;
  const scalar_t twiddle_val[2][2] = {{twiddle_a[s][twiddle_idx][0][0], twiddle_a[s][twiddle_idx][0][1]},
                                      {twiddle_a[s][twiddle_idx][1][0], twiddle_a[s][twiddle_idx][1][1]}};
  accscalar_t d_twiddle_val[2][2] = {{0, 0}, {0, 0}};
  for (int b = blockIdx.y * blockDim.y + threadIdx.y; b < batch_size; b += blockDim.y * gridDim.y) {
    const scalar_t grad_val[2] = {d_input_a[b][s][pos], d_input_a[b][s][pos + stride]};
    d_input_a[b][s][pos] = twiddle_val[0][0] * grad_val[0] + twiddle_val[1][0] * grad_val[1];
    d_input_a[b][s][pos + stride] = twiddle_val[0][1] * grad_val[0] + twiddle_val[1][1] * grad_val[1];
    const scalar_t input_val[2] = {output_a[idx][b][s][pos], output_a[idx][b][s][pos + stride]};
    d_twiddle_val[0][0] += grad_val[0] * input_val[0];
    d_twiddle_val[0][1] += grad_val[0] * input_val[1];
    d_twiddle_val[1][0] += grad_val[1] * input_val[0];
    d_twiddle_val[1][1] += grad_val[1] * input_val[1];
  }
  atomicAdd(&d_twiddle_a[s][twiddle_idx][0][0], d_twiddle_val[0][0]);
  atomicAdd(&d_twiddle_a[s][twiddle_idx][0][1], d_twiddle_val[0][1]);
  atomicAdd(&d_twiddle_a[s][twiddle_idx][1][0], d_twiddle_val[1][0]);
  atomicAdd(&d_twiddle_a[s][twiddle_idx][1][1], d_twiddle_val[1][1]);
}

template <typename scalar_t, typename accscalar_t, bool increasing_stride>
__global__ void butterfly_multiply_intermediate_backward_onestep_complex_cuda_kernel(const at::PackedTensorAccessor<scalar_t, 5> twiddle_a,
                                                                                     const at::PackedTensorAccessor<scalar_t, 5> output_a,
                                                                                     at::PackedTensorAccessor<scalar_t, 5> d_twiddle_a,
                                                                                     at::PackedTensorAccessor<scalar_t, 4> d_input_a,
                                                                                     int log_stride,
                                                                                     int log_n) {
  using complex_t = thrust::complex<scalar_t>;
  using acccomplex_t = thrust::complex<accscalar_t>;
  const int batch_size = output_a.size(1);
  const int s = blockIdx.z;
  const int idx = increasing_stride ? log_stride : (log_n - 1 - log_stride);  // Index to access output_a
  const int n = output_a.size(3);
  int stride = 1 << log_stride;
  int twiddle_start_idx = stride - 1;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i > n) return;
  int low_order_bits = i & (stride - 1);  // int low_order_bits = i % stride;
  int twiddle_idx = twiddle_start_idx + low_order_bits;
  int pos = 2 * (i - low_order_bits) + low_order_bits;
  const complex_t twiddle_val[2][2] =
    {{complex_t(twiddle_a[s][twiddle_idx][0][0][0], twiddle_a[s][twiddle_idx][0][0][1]),
      complex_t(twiddle_a[s][twiddle_idx][0][1][0], twiddle_a[s][twiddle_idx][0][1][1])},
     {complex_t(twiddle_a[s][twiddle_idx][1][0][0], twiddle_a[s][twiddle_idx][1][0][1]),
      complex_t(twiddle_a[s][twiddle_idx][1][1][0], twiddle_a[s][twiddle_idx][1][1][1])}};
  acccomplex_t d_twiddle_val[2][2] = {{{0, 0}, {0, 0}}, {{0, 0}, {0, 0}}};
  for (int b = blockIdx.y * blockDim.y + threadIdx.y; b < batch_size; b += blockDim.y * gridDim.y) {
    const complex_t grad_val[2] = {complex_t(d_input_a[b][s][pos][0], d_input_a[b][s][pos][1]),
                                   complex_t(d_input_a[b][s][pos + stride][0], d_input_a[b][s][pos + stride][1])};
    const complex_t d_input_val[2] =
      {thrust::conj(twiddle_val[0][0]) * grad_val[0] + thrust::conj(twiddle_val[1][0]) * grad_val[1],
       thrust::conj(twiddle_val[0][1]) * grad_val[0] + thrust::conj(twiddle_val[1][1]) * grad_val[1]};
    d_input_a[b][s][pos][0] = d_input_val[0].real();
    d_input_a[b][s][pos][1] = d_input_val[0].imag();
    d_input_a[b][s][pos + stride][0] = d_input_val[1].real();
    d_input_a[b][s][pos + stride][1] = d_input_val[1].imag();
    const complex_t input_val[2] =
      {complex_t(output_a[idx][b][s][pos][0], output_a[idx][b][s][pos][1]),
       complex_t(output_a[idx][b][s][pos + stride][0], output_a[idx][b][s][pos + stride][1])};
    d_twiddle_val[0][0] += grad_val[0] * thrust::conj(input_val[0]);
    d_twiddle_val[0][1] += grad_val[0] * thrust::conj(input_val[1]);
    d_twiddle_val[1][0] += grad_val[1] * thrust::conj(input_val[0]);
    d_twiddle_val[1][1] += grad_val[1] * thrust::conj(input_val[1]);
  }
  atomicAdd(&d_twiddle_a[s][twiddle_idx][0][0][0], d_twiddle_val[0][0].real());
  atomicAdd(&d_twiddle_a[s][twiddle_idx][0][0][1], d_twiddle_val[0][0].imag());
  atomicAdd(&d_twiddle_a[s][twiddle_idx][0][1][0], d_twiddle_val[0][1].real());
  atomicAdd(&d_twiddle_a[s][twiddle_idx][0][1][1], d_twiddle_val[0][1].imag());
  atomicAdd(&d_twiddle_a[s][twiddle_idx][1][0][0], d_twiddle_val[1][0].real());
  atomicAdd(&d_twiddle_a[s][twiddle_idx][1][0][1], d_twiddle_val[1][0].imag());
  atomicAdd(&d_twiddle_a[s][twiddle_idx][1][1][0], d_twiddle_val[1][1].real());
  atomicAdd(&d_twiddle_a[s][twiddle_idx][1][1][1], d_twiddle_val[1][1].imag());
}

void butterfly_multiply_intermediate_backward_cuda(const at::Tensor& twiddle, const at::Tensor& output,
                                                   at::Tensor& d_twiddle, at::Tensor& d_input, bool increasing_stride) {
  const int batch_size = output.size(1);
  const int nstack = output.size(2);
  const int n = output.size(3);
  const int log_n = int(log2((double) n));
  const bool complex = output.dim() == 5;
  AT_DISPATCH_FLOATING_TYPES(output.scalar_type(), "butterfly_multiply_intermediate_backward_cuda", [&] {
    using accscalar_t = at::acc_type<scalar_t, true>;
    if (!complex) {  // real
      const auto twiddle_a = twiddle.packed_accessor<scalar_t, 4>();
      const auto output_a = output.packed_accessor<scalar_t, 4>();
      auto d_twiddle_a = d_twiddle.packed_accessor<scalar_t, 4>();
      auto d_input_a = d_input.packed_accessor<scalar_t, 3>();
      if (increasing_stride) {
        int log_stride = log_n - 1;
        for (; (1 << log_stride) > ELEMENTARY_SIZE; --log_stride) {
          dim3 block(MAX_BLOCK_SIZE / 2);
          dim3 grid(div_up(n / 2, MAX_BLOCK_SIZE / 2), div_up(batch_size, WORK_PER_THREAD), nstack);
          butterfly_multiply_intermediate_backward_onestep_cuda_kernel<scalar_t, accscalar_t, true>
            <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, output_a, d_twiddle_a, d_input_a, log_stride, log_n);
        }
        int stride = 1 << log_stride;
        dim3 block(stride);
        dim3 grid(div_up(n / 2, stride), batch_size, nstack);
        butterfly_multiply_intermediate_backward_cuda_kernel<scalar_t, accscalar_t, true>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, output_a, d_twiddle_a, d_input_a, log_stride, log_n);
      } else {
        int stride = std::min<int>(ELEMENTARY_SIZE, n / 2);
        int log_stride = int(log2((double) stride));
        dim3 block(stride);
        dim3 grid(div_up(n / 2, stride), batch_size, nstack);
        butterfly_multiply_intermediate_backward_cuda_kernel<scalar_t, accscalar_t, false>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, output_a, d_twiddle_a, d_input_a, log_stride, log_n);
        for (log_stride++; log_stride <= log_n - 1; ++log_stride) {
          dim3 block(MAX_BLOCK_SIZE / 2);
          dim3 grid(div_up(n / 2, MAX_BLOCK_SIZE / 2), div_up(batch_size, WORK_PER_THREAD), nstack);
          butterfly_multiply_intermediate_backward_onestep_cuda_kernel<scalar_t, accscalar_t, false>
            <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, output_a, d_twiddle_a, d_input_a, log_stride, log_n);
        }
      }
    } else {  // complex
      const auto twiddle_a = twiddle.packed_accessor<scalar_t, 5>();
      const auto output_a = output.packed_accessor<scalar_t, 5>();
      auto d_twiddle_a = d_twiddle.packed_accessor<scalar_t, 5>();
      auto d_input_a = d_input.packed_accessor<scalar_t, 4>();
      if (increasing_stride) {
        int log_stride = log_n - 1;
        for (; (1 << log_stride) > ELEMENTARY_SIZE; --log_stride) {
          dim3 block(MAX_BLOCK_SIZE / 2);
          dim3 grid(div_up(n / 2, MAX_BLOCK_SIZE / 2), div_up(batch_size, WORK_PER_THREAD), nstack);
          butterfly_multiply_intermediate_backward_onestep_complex_cuda_kernel<scalar_t, accscalar_t, true>
            <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, output_a, d_twiddle_a, d_input_a, log_stride, log_n);
        }
        int stride = 1 << log_stride;
        dim3 block(stride);
        dim3 grid(div_up(n / 2, stride), batch_size, nstack);
        butterfly_multiply_intermediate_backward_complex_cuda_kernel<scalar_t, accscalar_t, true>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, output_a, d_twiddle_a, d_input_a, log_stride, log_n);
      } else {
        int stride = std::min<int>(ELEMENTARY_SIZE, n / 2);
        int log_stride = int(log2((double) stride));
        dim3 block(stride);
        dim3 grid(div_up(n / 2, stride), batch_size, nstack);
        butterfly_multiply_intermediate_backward_complex_cuda_kernel<scalar_t, accscalar_t, false>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, output_a, d_twiddle_a, d_input_a, log_stride, log_n);
        for (log_stride++; log_stride <= log_n - 1; ++log_stride) {
          dim3 block(MAX_BLOCK_SIZE / 2);
          dim3 grid(div_up(n / 2, MAX_BLOCK_SIZE / 2), div_up(batch_size, WORK_PER_THREAD), nstack);
          butterfly_multiply_intermediate_backward_onestep_complex_cuda_kernel<scalar_t, accscalar_t, false>
            <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, output_a, d_twiddle_a, d_input_a, log_stride, log_n);
        }
      }
    }
  });
  AT_CHECK(cudaGetLastError() == cudaSuccess,
     "butterfly_multiply_intermediate_backward_cuda failed with error code ",
     cudaGetLastError());
}

template <typename scalar_t, bool increasing_stride, bool return_intermediates>
__global__ void butterfly_multiply_untied_cuda_kernel(const at::PackedTensorAccessor<scalar_t, 5> twiddle_a,
                                                      at::PackedTensorAccessor<scalar_t, 4> output_a,
                                                      int log_max_stride,
                                                      int log_n) {
  const int batch_size = output_a.size(1);
  const int s = blockIdx.z;
  const int max_stride = 1 << log_max_stride;
  const int input_base_idx = blockIdx.y * blockDim.x * 2;
  __shared__ scalar_t s_input[ELEMENTARY_SIZE * 2];
  __shared__ scalar_t s_twiddle[ELEMENTARY_SIZE][2][2];
  int b = blockIdx.x * blockDim.y + threadIdx.y;
  int first_idx = increasing_stride ? 0 : log_n - 1 - log_max_stride;
  if (b < batch_size) {
    for (int i = threadIdx.x; i < max_stride * 2; i += blockDim.x) {
      s_input[i + threadIdx.y * max_stride * 2] = output_a[first_idx][b][s][input_base_idx + i];
    }
  }
  int tid_x = threadIdx.x;
  int tid_y = threadIdx.y;
  for (int idx = first_idx; idx <= first_idx + log_max_stride; ++idx) {
    int log_stride = increasing_stride ? idx : log_n - 1 - idx;
    int stride = 1 << log_stride;
    if (tid_y == 0) {
      s_twiddle[tid_x][0][0] = twiddle_a[s][log_stride][input_base_idx / 2 + tid_x][0][0];
      s_twiddle[tid_x][0][1] = twiddle_a[s][log_stride][input_base_idx / 2 + tid_x][0][1];
      s_twiddle[tid_x][1][0] = twiddle_a[s][log_stride][input_base_idx / 2 + tid_x][1][0];
      s_twiddle[tid_x][1][1] = twiddle_a[s][log_stride][input_base_idx / 2 + tid_x][1][1];
    }
    int low_order_bits = tid_x & (stride - 1);  // int low_order_bits = tid_x % stride;
    int pos_x = 2 * (tid_x - low_order_bits) + low_order_bits;
    int pos_y = tid_y * max_stride * 2;
    int pos = pos_x + pos_y;
    __syncthreads();
    const scalar_t twiddle_val[2][2] = {{s_twiddle[tid_x][0][0], s_twiddle[tid_x][0][1]},
                                        {s_twiddle[tid_x][1][0], s_twiddle[tid_x][1][1]}};
    __syncthreads();  // otherwise some thread might go back to writing to s_twiddle before other thread can read
    if (b < batch_size) {
      const scalar_t input_val[2] = {s_input[pos], s_input[pos + stride]};
      s_input[pos] = twiddle_val[0][0] * input_val[0] + twiddle_val[0][1] * input_val[1];
      s_input[pos + stride] = twiddle_val[1][0] * input_val[0] + twiddle_val[1][1] * input_val[1];
      if (return_intermediates || idx == first_idx + log_max_stride) {
        output_a[idx+1][b][s][input_base_idx + pos_x] = s_input[pos];
        output_a[idx+1][b][s][input_base_idx + pos_x + stride] = s_input[pos + stride];
      }
    }
  }
}

// Trying out an implementation where consecutive threads process same input index, but different batch indices.
// template <typename scalar_t, bool increasing_stride, bool return_intermediates>
// __global__ void butterfly_multiply_untied_cuda_kernel(const at::PackedTensorAccessor<scalar_t, 5> twiddle_a,
//                                                       at::PackedTensorAccessor<scalar_t, 4> output_a,
//                                                       int log_max_stride,
//                                                       int log_n) {
//   const int batch_size = output_a.size(1);
//   const int s = blockIdx.z;
//   const int max_stride = 1 << log_max_stride;
//   const int input_base_idx = blockIdx.y * blockDim.y * 2;
//   __shared__ scalar_t s_input[ELEMENTARY_SIZE * 2];
//   __shared__ scalar_t s_twiddle[ELEMENTARY_SIZE][2][2];
//   int b = blockIdx.x * blockDim.x + threadIdx.x;
//   int tid_x = threadIdx.x;  // batch index
//   int tid_y = threadIdx.y;
//   int first_idx = increasing_stride ? 0 : log_n - 1 - log_max_stride;
//   if (b < batch_size) {
//     for (int i = tid_y; i < max_stride * 2; i += blockDim.y) {
//       s_input[tid_x + i * blockDim.x] = output_a[first_idx][b][s][input_base_idx + i];
//     }
//   }
//   // for (int i = tid_x + tid_y * blockDim.x; i < blockDim.x * max_stride * 2; i += blockDim.x * blockDim.y) {
//   //   int input_idx = i & (max_stride * 2 - 1);  // int input_idx = i % (max_stride * 2);
//   //   int batch_idx = i >> (log_max_stride + 1);  // int batch_idx = (i - input_idx) / (max_stride * 2);
//   //   if (blockIdx.x * blockDim.x + batch_idx < batch_size) {
//   //     s_input[batch_idx + input_idx * blockDim.x] = output_a[blockIdx.x * blockDim.x + first_idx][batch_idx][s][input_base_idx + input_idx];
//   //   }
//   // }
//   for (int idx = first_idx; idx <= first_idx + log_max_stride; ++idx) {
//     int log_stride = increasing_stride ? idx : log_n - 1 - idx;
//     int stride = 1 << log_stride;
//     if (tid_x == 0) {
//       s_twiddle[tid_y][0][0] = twiddle_a[s][log_stride][input_base_idx / 2 + tid_y][0][0];
//       s_twiddle[tid_y][0][1] = twiddle_a[s][log_stride][input_base_idx / 2 + tid_y][0][1];
//       s_twiddle[tid_y][1][0] = twiddle_a[s][log_stride][input_base_idx / 2 + tid_y][1][0];
//       s_twiddle[tid_y][1][1] = twiddle_a[s][log_stride][input_base_idx / 2 + tid_y][1][1];
//     }
//     int low_order_bits = tid_y & (stride - 1);  // int low_order_bits = tid_y % stride;
//     int pos_y = 2 * (tid_y - low_order_bits) + low_order_bits;
//     int pos_x = tid_x;
//     int pos = pos_x + pos_y * blockDim.x;
//     __syncthreads();
//     const scalar_t twiddle_val[2][2] = {{s_twiddle[tid_y][0][0], s_twiddle[tid_y][0][1]},
//                                         {s_twiddle[tid_y][1][0], s_twiddle[tid_y][1][1]}};
//     __syncthreads();  // otherwise some thread might go back to writing to s_twiddle before other thread can read
//     if (b < batch_size) {
//       const scalar_t input_val[2] = {s_input[pos], s_input[pos + stride * blockDim.x]};
//       s_input[pos] = twiddle_val[0][0] * input_val[0] + twiddle_val[0][1] * input_val[1];
//       s_input[pos + stride * blockDim.x] = twiddle_val[1][0] * input_val[0] + twiddle_val[1][1] * input_val[1];
//       if (return_intermediates || idx == first_idx + log_max_stride) {
//         output_a[idx+1][b][s][input_base_idx + pos_y] = s_input[pos];
//         output_a[idx+1][b][s][input_base_idx + pos_y + stride] = s_input[pos + stride * blockDim.x];
//       }
//     }
//   }
// }

template <typename scalar_t, bool increasing_stride, bool return_intermediates>
__global__ void butterfly_multiply_untied_complex_cuda_kernel(const at::PackedTensorAccessor<scalar_t, 6> twiddle_a,
                                                              at::PackedTensorAccessor<scalar_t, 5> output_a,
                                                              int log_max_stride,
                                                              int log_n) {
  using complex_t = thrust::complex<scalar_t>;
  const int batch_size = output_a.size(1);
  const int s = blockIdx.z;
  const int max_stride = 1 << log_max_stride;
  const int input_base_idx = blockIdx.y * blockDim.x * 2;
  // __shared__ complex_t s_input[ELEMENTARY_SIZE * 2];
  __shared__ scalar_t s_input_storage[ELEMENTARY_SIZE * 2][2];
  complex_t* s_input = (complex_t *)&s_input_storage[0];  // To avoid warning about race-condition when initializing complex_t
  int b = blockIdx.x * blockDim.y + threadIdx.y;
  if (b < batch_size) {  // Currently we assume 1 batch per thread block, so all threads in the block should enter (otherwise deadlock)
    int first_idx = increasing_stride ? 0 : log_n - 1 - log_max_stride;
    for (int i = threadIdx.x; i < max_stride * 2; i += blockDim.x) {
      s_input[i] = complex_t(output_a[first_idx][b][s][input_base_idx + i][0], output_a[first_idx][b][s][input_base_idx + i][1]);
    }
    int i = threadIdx.x;
    for (int idx = first_idx; idx <= first_idx + log_max_stride; ++idx) {
      int log_stride = increasing_stride ? idx : log_n - 1 - idx;
      int stride = 1 << log_stride;
      int low_order_bits = i & (stride - 1);  // int low_order_bits = i % stride;
      int pos = 2 * (i - low_order_bits) + low_order_bits;
      const complex_t twiddle_val[2][2] =
        {{complex_t(twiddle_a[s][log_stride][input_base_idx / 2 + i][0][0][0], twiddle_a[s][log_stride][input_base_idx / 2 + i][0][0][1]),
          complex_t(twiddle_a[s][log_stride][input_base_idx / 2 + i][0][1][0], twiddle_a[s][log_stride][input_base_idx / 2 + i][0][1][1])},
         {complex_t(twiddle_a[s][log_stride][input_base_idx / 2 + i][1][0][0], twiddle_a[s][log_stride][input_base_idx / 2 + i][1][0][1]),
          complex_t(twiddle_a[s][log_stride][input_base_idx / 2 + i][1][1][0], twiddle_a[s][log_stride][input_base_idx / 2 + i][1][1][1])}};
      __syncthreads();
      const complex_t input_val[2] = {s_input[pos], s_input[pos + stride]};
      s_input[pos] = twiddle_val[0][0] * input_val[0] + twiddle_val[0][1] * input_val[1];
      s_input[pos + stride] = twiddle_val[1][0] * input_val[0] + twiddle_val[1][1] * input_val[1];
      if (return_intermediates || idx == first_idx + log_max_stride) {
        output_a[idx+1][b][s][input_base_idx + pos][0] = s_input[pos].real();
        output_a[idx+1][b][s][input_base_idx + pos][1] = s_input[pos].imag();
        output_a[idx+1][b][s][input_base_idx + pos + stride][0] = s_input[pos + stride].real();
        output_a[idx+1][b][s][input_base_idx + pos + stride][1] = s_input[pos + stride].imag();
      }
    }
  }
}

template <typename scalar_t, bool increasing_stride>
__global__ void butterfly_multiply_untied_onestep_cuda_kernel(const at::PackedTensorAccessor<scalar_t, 5> twiddle_a,
                                                              at::PackedTensorAccessor<scalar_t, 4> output_a,
                                                              int log_stride,
                                                              int log_n) {
  const int batch_size = output_a.size(1);
  const int s = blockIdx.z;
  const int idx = increasing_stride ? log_stride : (log_n - 1 - log_stride);  // Index to access output_a
  const int stride = 1 << log_stride;
  int i = blockIdx.y * blockDim.x + threadIdx.x;
  int low_order_bits = i & (stride - 1);  // int low_order_bits = i % stride;
  int pos = 2 * (i - low_order_bits) + low_order_bits;
  const scalar_t twiddle_val[2][2] = {{twiddle_a[s][log_stride][i][0][0], twiddle_a[s][log_stride][i][0][1]},
                                      {twiddle_a[s][log_stride][i][1][0], twiddle_a[s][log_stride][i][1][1]}};
  for (int b = blockIdx.x * blockDim.y + threadIdx.y; b < batch_size; b += blockDim.y * gridDim.x) {
    const scalar_t input_val[2] = {output_a[idx][b][s][pos], output_a[idx][b][s][pos + stride]};
    output_a[idx+1][b][s][pos] = twiddle_val[0][0] * input_val[0] + twiddle_val[0][1] * input_val[1];
    output_a[idx+1][b][s][pos + stride] = twiddle_val[1][0] * input_val[0] + twiddle_val[1][1] * input_val[1];
  }
}

template <typename scalar_t, bool increasing_stride>
__global__ void butterfly_multiply_untied_onestep_complex_cuda_kernel(const at::PackedTensorAccessor<scalar_t, 6> twiddle_a,
                                                                      at::PackedTensorAccessor<scalar_t, 5> output_a,
                                                                      int log_stride,
                                                                      int log_n) {
  using complex_t = thrust::complex<scalar_t>;
  const int batch_size = output_a.size(1);
  const int s = blockIdx.z;
  const int idx = increasing_stride ? log_stride : (log_n - 1 - log_stride);  // Index to access output_a
  const int stride = 1 << log_stride;
  int i = blockIdx.y * blockDim.x + threadIdx.x;
  int low_order_bits = i & (stride - 1);  // int low_order_bits = i % stride;
  int pos = 2 * (i - low_order_bits) + low_order_bits;
  const complex_t twiddle_val[2][2] =
    {{complex_t(twiddle_a[s][log_stride][i][0][0][0], twiddle_a[s][log_stride][i][0][0][1]),
      complex_t(twiddle_a[s][log_stride][i][0][1][0], twiddle_a[s][log_stride][i][0][1][1])},
     {complex_t(twiddle_a[s][log_stride][i][1][0][0], twiddle_a[s][log_stride][i][1][0][1]),
      complex_t(twiddle_a[s][log_stride][i][1][1][0], twiddle_a[s][log_stride][i][1][1][1])}};
  for (int b = blockIdx.x * blockDim.y + threadIdx.y; b < batch_size; b += blockDim.y * gridDim.x) {
    const complex_t input_val[2] =
      {complex_t(output_a[idx][b][s][pos][0], output_a[idx][b][s][pos][1]),
       complex_t(output_a[idx][b][s][pos + stride][0], output_a[idx][b][s][pos + stride][1])};
    const complex_t output_val[2] =
      {twiddle_val[0][0] * input_val[0] + twiddle_val[0][1] * input_val[1],
       twiddle_val[1][0] * input_val[0] + twiddle_val[1][1] * input_val[1]};
    output_a[idx+1][b][s][pos][0] = output_val[0].real();
    output_a[idx+1][b][s][pos][1] = output_val[0].imag();
    output_a[idx+1][b][s][pos + stride][0] = output_val[1].real();
    output_a[idx+1][b][s][pos + stride][1] = output_val[1].imag();
  }
}

void butterfly_multiply_untied_cuda(const at::Tensor& twiddle, at::Tensor& output, bool increasing_stride, bool return_intermediates) {
  const int batch_size = output.size(1);
  const int nstack = twiddle.size(0);
  const int n = output.size(3);
  const int log_n = int(log2((double) n));
  const bool complex = output.dim() == 5;
  AT_DISPATCH_FLOATING_TYPES(output.scalar_type(), "butterfly_multiply_untied_cuda", [&] {
    if (!complex) {  // real
      const auto twiddle_a = twiddle.packed_accessor<scalar_t, 5>();
      auto output_a = output.packed_accessor<scalar_t, 4>();
      if (increasing_stride) {
        int stride = std::min<int>(ELEMENTARY_SIZE, n / 2);
        int log_stride = int(log2((double) stride));
        dim3 block(stride, div_up(MAX_BLOCK_SIZE, stride * 2));
        dim3 grid(div_up(batch_size, block.y), div_up(n / 2, stride), nstack);
        // dim3 block(div_up(MAX_BLOCK_SIZE, stride * 2), stride);
        // dim3 grid(div_up(batch_size, block.x), div_up(n / 2, stride), nstack);
        return_intermediates ? butterfly_multiply_untied_cuda_kernel<scalar_t, true, true>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, output_a, log_stride, log_n)
                             : butterfly_multiply_untied_cuda_kernel<scalar_t, true, false>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, output_a, log_stride, log_n);
        for (log_stride++; log_stride <= log_n - 1; ++log_stride) {
          dim3 block(MAX_BLOCK_SIZE / 2);
          dim3 grid(div_up(batch_size, WORK_PER_THREAD), div_up(n / 2, MAX_BLOCK_SIZE / 2), nstack);
          butterfly_multiply_untied_onestep_cuda_kernel<scalar_t, true>
            <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, output_a, log_stride, log_n);
        }
      } else {
        int log_stride = log_n - 1;
        for (; (1 << log_stride) > ELEMENTARY_SIZE; --log_stride) {
          dim3 block(MAX_BLOCK_SIZE / 2);
          dim3 grid(div_up(batch_size, WORK_PER_THREAD), div_up(n / 2, MAX_BLOCK_SIZE / 2), nstack);
          butterfly_multiply_untied_onestep_cuda_kernel<scalar_t, false>
            <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, output_a, log_stride, log_n);
        }
        int stride = 1 << log_stride;
        dim3 block(stride, div_up(MAX_BLOCK_SIZE, stride * 2));
        dim3 grid(div_up(batch_size, block.y), div_up(n / 2, stride), nstack);
        // dim3 block(div_up(MAX_BLOCK_SIZE, stride * 2), stride);
        // dim3 grid(div_up(batch_size, block.x), div_up(n / 2, stride), nstack);
        return_intermediates ? butterfly_multiply_untied_cuda_kernel<scalar_t, false, true>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, output_a, log_stride, log_n)
                             : butterfly_multiply_untied_cuda_kernel<scalar_t, false, false>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, output_a, log_stride, log_n);
      }
    } else {  // complex
      const auto twiddle_a = twiddle.packed_accessor<scalar_t, 6>();
      auto output_a = output.packed_accessor<scalar_t, 5>();
      if (increasing_stride) {
        int stride = std::min<int>(ELEMENTARY_SIZE, n / 2);
        int log_stride = int(log2((double) stride));
        dim3 block(stride);
        dim3 grid(batch_size, div_up(n / 2, stride), nstack);
        return_intermediates ? butterfly_multiply_untied_complex_cuda_kernel<scalar_t, true, true>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, output_a, log_stride, log_n)
                             : butterfly_multiply_untied_complex_cuda_kernel<scalar_t, true, false>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, output_a, log_stride, log_n);
        for (log_stride++; log_stride <= log_n - 1; ++log_stride) {
          dim3 block(MAX_BLOCK_SIZE / 2);
          dim3 grid(div_up(batch_size, WORK_PER_THREAD), div_up(n / 2, MAX_BLOCK_SIZE / 2), nstack);
          butterfly_multiply_untied_onestep_complex_cuda_kernel<scalar_t, true>
            <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, output_a, log_stride, log_n);
        }
      } else {
        int log_stride = log_n - 1;
        for (; (1 << log_stride) > ELEMENTARY_SIZE; --log_stride) {
          dim3 block(MAX_BLOCK_SIZE / 2);
          dim3 grid(div_up(batch_size, WORK_PER_THREAD), div_up(n / 2, MAX_BLOCK_SIZE / 2), nstack);
          butterfly_multiply_untied_onestep_complex_cuda_kernel<scalar_t, false>
            <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, output_a, log_stride, log_n);
        }
        int stride = 1 << log_stride;
        dim3 block(stride);
        dim3 grid(batch_size, div_up(n / 2, stride), nstack);
        return_intermediates ? butterfly_multiply_untied_complex_cuda_kernel<scalar_t, false, true>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, output_a, log_stride, log_n)
                             : butterfly_multiply_untied_complex_cuda_kernel<scalar_t, false, false>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, output_a, log_stride, log_n);
      }
    }
  });
  AT_CHECK(cudaGetLastError() == cudaSuccess,
     "butterfly_multiply_untied_cuda failed with error code ",
     cudaGetLastError());
}

// Original implementation, with 1 batch per thread block
// template <typename scalar_t, typename accscalar_t, bool increasing_stride>
// __global__ void butterfly_multiply_untied_backward_cuda_kernel(const at::PackedTensorAccessor<scalar_t, 5> twiddle_a,
//                                                                const at::PackedTensorAccessor<scalar_t, 4> output_a,
//                                                                at::PackedTensorAccessor<scalar_t, 5> d_twiddle_a,
//                                                                at::PackedTensorAccessor<scalar_t, 3> d_input_a,
//                                                                int log_max_stride,
//                                                                int log_n) {
//   const int batch_size = output_a.size(1);
//   const int s = blockIdx.z;
//   const int max_stride = 1 << log_max_stride;
//   const int input_base_idx = blockIdx.y * blockDim.x * 2;
//   __shared__ scalar_t s_grad[ELEMENTARY_SIZE * 2];
//   int b = blockIdx.x * blockDim.y + threadIdx.y;
//   if (b < batch_size) {  // Currently we assume 1 batch per thread block, so all threads in the block should enter (otherwise deadlock)
//     for (int i = threadIdx.x; i < max_stride * 2; i += blockDim.x) {
//       s_grad[i] = d_input_a[b][s][input_base_idx + i];
//     }
//     int i = threadIdx.x;
//     int first_idx = increasing_stride ? 0 : log_n - 1 - log_max_stride;
//     for (int idx = first_idx + log_max_stride; idx >= first_idx; --idx) {
//       int log_stride = increasing_stride ? idx : log_n - 1 - idx;
//       int stride = 1 << log_stride;
//       int low_order_bits = i & (stride - 1);  // int low_order_bits = i % stride;
//       int pos = 2 * (i - low_order_bits) + low_order_bits;
//       const scalar_t twiddle_val[2][2] = {{twiddle_a[s][log_stride][input_base_idx / 2 + i][0][0], twiddle_a[s][log_stride][input_base_idx / 2 + i][0][1]},
//                                           {twiddle_a[s][log_stride][input_base_idx / 2 + i][1][0], twiddle_a[s][log_stride][input_base_idx / 2 + i][1][1]}};
//       __syncthreads();
//       const scalar_t grad_val[2] = {s_grad[pos], s_grad[pos + stride]};
//       s_grad[pos] = twiddle_val[0][0] * grad_val[0] + twiddle_val[1][0] * grad_val[1];
//       s_grad[pos + stride] = twiddle_val[0][1] * grad_val[0] + twiddle_val[1][1] * grad_val[1];
//       const scalar_t input_val[2] = {output_a[idx][b][s][input_base_idx + pos], output_a[idx][b][s][input_base_idx + pos + stride]};
//       accscalar_t d_twiddle_val[2][2] = {{grad_val[0] * input_val[0], grad_val[0] * input_val[1]},
//                                          {grad_val[1] * input_val[0], grad_val[1] * input_val[1]}};
//       atomicAdd(&d_twiddle_a[s][log_stride][input_base_idx / 2 + i][0][0], d_twiddle_val[0][0]);
//       atomicAdd(&d_twiddle_a[s][log_stride][input_base_idx / 2 + i][0][1], d_twiddle_val[0][1]);
//       atomicAdd(&d_twiddle_a[s][log_stride][input_base_idx / 2 + i][1][0], d_twiddle_val[1][0]);
//       atomicAdd(&d_twiddle_a[s][log_stride][input_base_idx / 2 + i][1][1], d_twiddle_val[1][1]);
//     }
//     __syncthreads();
//     for (int i = threadIdx.x; i < max_stride * 2; i += blockDim.x) {
//       d_input_a[b][s][input_base_idx + i] = s_grad[i];
//     }
//   }
// }

template <typename scalar_t, typename accscalar_t, bool increasing_stride>
__global__ void butterfly_multiply_untied_backward_cuda_kernel(const at::PackedTensorAccessor<scalar_t, 5> twiddle_a,
                                                               const at::PackedTensorAccessor<scalar_t, 4> output_a,
                                                               at::PackedTensorAccessor<scalar_t, 5> d_twiddle_a,
                                                               at::PackedTensorAccessor<scalar_t, 3> d_input_a,
                                                               int log_max_stride,
                                                               int log_n) {
  const int batch_size = output_a.size(1);
  const int s = blockIdx.z;
  const int max_stride = 1 << log_max_stride;
  const int input_base_idx = blockIdx.y * blockDim.x * 2;
  __shared__ scalar_t s_grad[ELEMENTARY_SIZE * 2];
  __shared__ accscalar_t s_twiddle[ELEMENTARY_SIZE][2][2];  // Use accscalar_t instead of scalar_t since we'll reuse the storage for s_d_twiddle
  accscalar_t* s_d_twiddle = (accscalar_t *)&s_twiddle[0][0][0];  // Reusing the same storage as s_twiddle, have to be careful if we change the implemetnation.
  int b = blockIdx.x * blockDim.y + threadIdx.y;
  if (b < batch_size) {
    for (int i = threadIdx.x; i < max_stride * 2; i += blockDim.x) {
      s_grad[i + threadIdx.y * max_stride * 2] = d_input_a[b][s][input_base_idx + i];
    }
  }
  int tid_x = threadIdx.x;
  int tid_y = threadIdx.y;
  int first_idx = increasing_stride ? 0 : log_n - 1 - log_max_stride;
  for (int idx = first_idx + log_max_stride; idx >= first_idx; --idx) {
    int log_stride = increasing_stride ? idx : log_n - 1 - idx;
    int stride = 1 << log_stride;
    if (tid_y == 0) {
      s_twiddle[tid_x][0][0] = twiddle_a[s][log_stride][input_base_idx / 2 + tid_x][0][0];
      s_twiddle[tid_x][0][1] = twiddle_a[s][log_stride][input_base_idx / 2 + tid_x][0][1];
      s_twiddle[tid_x][1][0] = twiddle_a[s][log_stride][input_base_idx / 2 + tid_x][1][0];
      s_twiddle[tid_x][1][1] = twiddle_a[s][log_stride][input_base_idx / 2 + tid_x][1][1];
    }
    int low_order_bits = tid_x & (stride - 1);  // int low_order_bits = tid_x % stride;
    int pos_x = 2 * (tid_x - low_order_bits) + low_order_bits;
    int pos_y = tid_y * max_stride * 2;
    int pos = pos_x + pos_y;
    __syncthreads();
    const scalar_t twiddle_val[2][2] = {{s_twiddle[tid_x][0][0], s_twiddle[tid_x][0][1]},
                                        {s_twiddle[tid_x][1][0], s_twiddle[tid_x][1][1]}};
    // Don't need to sync here since we sync later at sum_strided_atomic, so no writing to s_twiddle can occur until then
    accscalar_t d_twiddle_val[2][2] = {{0, 0}, {0, 0}};
    if (b < batch_size) {
      const scalar_t grad_val[2] = {s_grad[pos], s_grad[pos + stride]};
      s_grad[pos] = twiddle_val[0][0] * grad_val[0] + twiddle_val[1][0] * grad_val[1];
      s_grad[pos + stride] = twiddle_val[0][1] * grad_val[0] + twiddle_val[1][1] * grad_val[1];
      const scalar_t input_val[2] = {output_a[idx][b][s][input_base_idx + pos_x],
                                     output_a[idx][b][s][input_base_idx + pos_x + stride]};
      d_twiddle_val[0][0] = grad_val[0] * input_val[0];
      d_twiddle_val[0][1] = grad_val[0] * input_val[1];
      d_twiddle_val[1][0] = grad_val[1] * input_val[0];
      d_twiddle_val[1][1] = grad_val[1] * input_val[1];
    }
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int nthreads = blockDim.x * blockDim.y;
    sum_strided_atomic(reinterpret_cast<accscalar_t (&)[4]>(d_twiddle_val), s_d_twiddle, max_stride, nthreads, tid);
    if (tid_y == 0) {
      atomicAdd(&d_twiddle_a[s][log_stride][input_base_idx / 2 + tid_x][0][0], s_d_twiddle[tid_x]);
      atomicAdd(&d_twiddle_a[s][log_stride][input_base_idx / 2 + tid_x][0][1], s_d_twiddle[tid_x + max_stride]);
      atomicAdd(&d_twiddle_a[s][log_stride][input_base_idx / 2 + tid_x][1][0], s_d_twiddle[tid_x + 2 * max_stride]);
      atomicAdd(&d_twiddle_a[s][log_stride][input_base_idx / 2 + tid_x][1][1], s_d_twiddle[tid_x + 3 * max_stride]);
    }
    __syncthreads();  // Otherwise s_d_twiddle will be overwritten with s_twiddle before some thread can read
  }
  if (b < batch_size) {
    for (int i = threadIdx.x; i < max_stride * 2; i += blockDim.x) {
      d_input_a[b][s][input_base_idx + i] = s_grad[i + threadIdx.y * max_stride * 2];
    }
  }
}

template <typename scalar_t, typename accscalar_t, bool increasing_stride>
__global__ void butterfly_multiply_untied_backward_complex_cuda_kernel(const at::PackedTensorAccessor<scalar_t, 6> twiddle_a,
                                                                       const at::PackedTensorAccessor<scalar_t, 5> output_a,
                                                                       at::PackedTensorAccessor<scalar_t, 6> d_twiddle_a,
                                                                       at::PackedTensorAccessor<scalar_t, 4> d_input_a,
                                                                       int log_max_stride,
                                                                       int log_n) {
  using complex_t = thrust::complex<scalar_t>;
  using acccomplex_t = thrust::complex<accscalar_t>;
  const int batch_size = output_a.size(1);
  const int s = blockIdx.z;
  const int max_stride = 1 << log_max_stride;
  const int input_base_idx = blockIdx.y * blockDim.x * 2;
  // __shared__ scalar_t s_grad[ELEMENTARY_SIZE * 2][2];
  __shared__ scalar_t s_grad_storage[ELEMENTARY_SIZE * 2][2];
  complex_t* s_grad = (complex_t *)&s_grad_storage[0];  // To avoid warning about race-condition when initializing complex_t
  int b = blockIdx.x * blockDim.y + threadIdx.y;
  if (b < batch_size) {  // Currently we assume 1 batch per thread block, so all threads in the block should enter (otherwise deadlock)
    for (int i = threadIdx.x; i < max_stride * 2; i += blockDim.x) {
      s_grad[i] = complex_t(d_input_a[b][s][input_base_idx + i][0], d_input_a[b][s][input_base_idx + i][1]);
    }
    int i = threadIdx.x;
    int first_idx = increasing_stride ? 0 : log_n - 1 - log_max_stride;
    for (int idx = first_idx + log_max_stride; idx >= first_idx; --idx) {
      int log_stride = increasing_stride ? idx : log_n - 1 - idx;
      int stride = 1 << log_stride;
      int low_order_bits = i & (stride - 1);  // int low_order_bits = i % stride;
      int pos = 2 * (i - low_order_bits) + low_order_bits;
      const complex_t twiddle_val[2][2] =
        {{complex_t(twiddle_a[s][log_stride][input_base_idx / 2 + i][0][0][0], twiddle_a[s][log_stride][input_base_idx / 2 + i][0][0][1]),
          complex_t(twiddle_a[s][log_stride][input_base_idx / 2 + i][0][1][0], twiddle_a[s][log_stride][input_base_idx / 2 + i][0][1][1])},
         {complex_t(twiddle_a[s][log_stride][input_base_idx / 2 + i][1][0][0], twiddle_a[s][log_stride][input_base_idx / 2 + i][1][0][1]),
          complex_t(twiddle_a[s][log_stride][input_base_idx / 2 + i][1][1][0], twiddle_a[s][log_stride][input_base_idx / 2 + i][1][1][1])}};
      __syncthreads();
      const complex_t grad_val[2] = {s_grad[pos], s_grad[pos + stride]};
      s_grad[pos] = thrust::conj(twiddle_val[0][0]) * grad_val[0] + thrust::conj(twiddle_val[1][0]) * grad_val[1];
      s_grad[pos + stride] = thrust::conj(twiddle_val[0][1]) * grad_val[0] + thrust::conj(twiddle_val[1][1]) * grad_val[1];
      const complex_t input_val[2] =
        {complex_t(output_a[idx][b][s][input_base_idx + pos][0], output_a[idx][b][s][input_base_idx + pos][1]),
         complex_t(output_a[idx][b][s][input_base_idx + pos + stride][0], output_a[idx][b][s][input_base_idx + pos + stride][1])};
      acccomplex_t d_twiddle_val[2][2] =
        {{grad_val[0] * thrust::conj(input_val[0]), grad_val[0] * thrust::conj(input_val[1])},
         {grad_val[1] * thrust::conj(input_val[0]), grad_val[1] * thrust::conj(input_val[1])}};
      atomicAdd(&d_twiddle_a[s][log_stride][input_base_idx / 2 + i][0][0][0], d_twiddle_val[0][0].real());
      atomicAdd(&d_twiddle_a[s][log_stride][input_base_idx / 2 + i][0][0][1], d_twiddle_val[0][0].imag());
      atomicAdd(&d_twiddle_a[s][log_stride][input_base_idx / 2 + i][0][1][0], d_twiddle_val[0][1].real());
      atomicAdd(&d_twiddle_a[s][log_stride][input_base_idx / 2 + i][0][1][1], d_twiddle_val[0][1].imag());
      atomicAdd(&d_twiddle_a[s][log_stride][input_base_idx / 2 + i][1][0][0], d_twiddle_val[1][0].real());
      atomicAdd(&d_twiddle_a[s][log_stride][input_base_idx / 2 + i][1][0][1], d_twiddle_val[1][0].imag());
      atomicAdd(&d_twiddle_a[s][log_stride][input_base_idx / 2 + i][1][1][0], d_twiddle_val[1][1].real());
      atomicAdd(&d_twiddle_a[s][log_stride][input_base_idx / 2 + i][1][1][1], d_twiddle_val[1][1].imag());
    }
    __syncthreads();
    for (int i = threadIdx.x; i < max_stride * 2; i += blockDim.x) {
      d_input_a[b][s][input_base_idx + i][0] = s_grad[i].real();
      d_input_a[b][s][input_base_idx + i][1] = s_grad[i].imag();
    }
  }
}

template <typename scalar_t, typename accscalar_t, bool increasing_stride>
__global__ void butterfly_multiply_untied_backward_onestep_cuda_kernel(const at::PackedTensorAccessor<scalar_t, 5> twiddle_a,
                                                                       const at::PackedTensorAccessor<scalar_t, 4> output_a,
                                                                       at::PackedTensorAccessor<scalar_t, 5> d_twiddle_a,
                                                                       at::PackedTensorAccessor<scalar_t, 3> d_input_a,
                                                                       int log_stride,
                                                                       int log_n) {
  const int batch_size = output_a.size(1);
  const int s = blockIdx.z;
  const int idx = increasing_stride ? log_stride : (log_n - 1 - log_stride);  // Index to access output_a
  const int n = output_a.size(3);
  int stride = 1 << log_stride;
  int i = blockIdx.y * blockDim.x + threadIdx.x;
  if (i > n) return;
  int low_order_bits = i & (stride - 1);  // int low_order_bits = i % stride;
  int pos = 2 * (i - low_order_bits) + low_order_bits;
  const scalar_t twiddle_val[2][2] = {{twiddle_a[s][log_stride][i][0][0], twiddle_a[s][log_stride][i][0][1]},
                                      {twiddle_a[s][log_stride][i][1][0], twiddle_a[s][log_stride][i][1][1]}};
  accscalar_t d_twiddle_val[2][2] = {{0, 0}, {0, 0}};
  for (int b = blockIdx.x * blockDim.y + threadIdx.y; b < batch_size; b += blockDim.y * gridDim.x) {
    const scalar_t grad_val[2] = {d_input_a[b][s][pos], d_input_a[b][s][pos + stride]};
    d_input_a[b][s][pos] = twiddle_val[0][0] * grad_val[0] + twiddle_val[1][0] * grad_val[1];
    d_input_a[b][s][pos + stride] = twiddle_val[0][1] * grad_val[0] + twiddle_val[1][1] * grad_val[1];
    const scalar_t input_val[2] = {output_a[idx][b][s][pos], output_a[idx][b][s][pos + stride]};
    d_twiddle_val[0][0] += grad_val[0] * input_val[0];
    d_twiddle_val[0][1] += grad_val[0] * input_val[1];
    d_twiddle_val[1][0] += grad_val[1] * input_val[0];
    d_twiddle_val[1][1] += grad_val[1] * input_val[1];
  }
  atomicAdd(&d_twiddle_a[s][log_stride][i][0][0], d_twiddle_val[0][0]);
  atomicAdd(&d_twiddle_a[s][log_stride][i][0][1], d_twiddle_val[0][1]);
  atomicAdd(&d_twiddle_a[s][log_stride][i][1][0], d_twiddle_val[1][0]);
  atomicAdd(&d_twiddle_a[s][log_stride][i][1][1], d_twiddle_val[1][1]);
}

template <typename scalar_t, typename accscalar_t, bool increasing_stride>
__global__ void butterfly_multiply_untied_backward_onestep_complex_cuda_kernel(const at::PackedTensorAccessor<scalar_t, 6> twiddle_a,
                                                                               const at::PackedTensorAccessor<scalar_t, 5> output_a,
                                                                               at::PackedTensorAccessor<scalar_t, 6> d_twiddle_a,
                                                                               at::PackedTensorAccessor<scalar_t, 4> d_input_a,
                                                                               int log_stride,
                                                                               int log_n) {
  using complex_t = thrust::complex<scalar_t>;
  using acccomplex_t = thrust::complex<accscalar_t>;
  const int batch_size = output_a.size(1);
  const int s = blockIdx.z;
  const int idx = increasing_stride ? log_stride : (log_n - 1 - log_stride);  // Index to access output_a
  const int n = output_a.size(3);
  int stride = 1 << log_stride;
  int i = blockIdx.y * blockDim.x + threadIdx.x;
  if (i > n) return;
  int low_order_bits = i & (stride - 1);  // int low_order_bits = i % stride;
  int pos = 2 * (i - low_order_bits) + low_order_bits;
  const complex_t twiddle_val[2][2] =
    {{complex_t(twiddle_a[s][log_stride][i][0][0][0], twiddle_a[s][log_stride][i][0][0][1]),
      complex_t(twiddle_a[s][log_stride][i][0][1][0], twiddle_a[s][log_stride][i][0][1][1])},
     {complex_t(twiddle_a[s][log_stride][i][1][0][0], twiddle_a[s][log_stride][i][1][0][1]),
      complex_t(twiddle_a[s][log_stride][i][1][1][0], twiddle_a[s][log_stride][i][1][1][1])}};
  acccomplex_t d_twiddle_val[2][2] = {{{0, 0}, {0, 0}}, {{0, 0}, {0, 0}}};
  for (int b = blockIdx.x * blockDim.y + threadIdx.y; b < batch_size; b += blockDim.y * gridDim.x) {
    const complex_t grad_val[2] = {complex_t(d_input_a[b][s][pos][0], d_input_a[b][s][pos][1]),
                                   complex_t(d_input_a[b][s][pos + stride][0], d_input_a[b][s][pos + stride][1])};
    const complex_t d_input_val[2] =
      {thrust::conj(twiddle_val[0][0]) * grad_val[0] + thrust::conj(twiddle_val[1][0]) * grad_val[1],
       thrust::conj(twiddle_val[0][1]) * grad_val[0] + thrust::conj(twiddle_val[1][1]) * grad_val[1]};
    d_input_a[b][s][pos][0] = d_input_val[0].real();
    d_input_a[b][s][pos][1] = d_input_val[0].imag();
    d_input_a[b][s][pos + stride][0] = d_input_val[1].real();
    d_input_a[b][s][pos + stride][1] = d_input_val[1].imag();
    const complex_t input_val[2] =
      {complex_t(output_a[idx][b][s][pos][0], output_a[idx][b][s][pos][1]),
       complex_t(output_a[idx][b][s][pos + stride][0], output_a[idx][b][s][pos + stride][1])};
    d_twiddle_val[0][0] += grad_val[0] * thrust::conj(input_val[0]);
    d_twiddle_val[0][1] += grad_val[0] * thrust::conj(input_val[1]);
    d_twiddle_val[1][0] += grad_val[1] * thrust::conj(input_val[0]);
    d_twiddle_val[1][1] += grad_val[1] * thrust::conj(input_val[1]);
  }
  atomicAdd(&d_twiddle_a[s][log_stride][i][0][0][0], d_twiddle_val[0][0].real());
  atomicAdd(&d_twiddle_a[s][log_stride][i][0][0][1], d_twiddle_val[0][0].imag());
  atomicAdd(&d_twiddle_a[s][log_stride][i][0][1][0], d_twiddle_val[0][1].real());
  atomicAdd(&d_twiddle_a[s][log_stride][i][0][1][1], d_twiddle_val[0][1].imag());
  atomicAdd(&d_twiddle_a[s][log_stride][i][1][0][0], d_twiddle_val[1][0].real());
  atomicAdd(&d_twiddle_a[s][log_stride][i][1][0][1], d_twiddle_val[1][0].imag());
  atomicAdd(&d_twiddle_a[s][log_stride][i][1][1][0], d_twiddle_val[1][1].real());
  atomicAdd(&d_twiddle_a[s][log_stride][i][1][1][1], d_twiddle_val[1][1].imag());
}

void butterfly_multiply_untied_backward_cuda(const at::Tensor& twiddle, const at::Tensor& output,
                                             at::Tensor& d_twiddle, at::Tensor& d_input, bool increasing_stride) {
  const int batch_size = output.size(1);
  const int nstack = output.size(2);
  const int n = output.size(3);
  const int log_n = int(log2((double) n));
  const bool complex = output.dim() == 5;
  AT_DISPATCH_FLOATING_TYPES(output.scalar_type(), "butterfly_multiply_untied_backward_cuda", [&] {
    using accscalar_t = at::acc_type<scalar_t, true>;
    if (!complex) {  // real
      const auto twiddle_a = twiddle.packed_accessor<scalar_t, 5>();
      const auto output_a = output.packed_accessor<scalar_t, 4>();
      auto d_twiddle_a = d_twiddle.packed_accessor<scalar_t, 5>();
      auto d_input_a = d_input.packed_accessor<scalar_t, 3>();
      if (increasing_stride) {
        int log_stride = log_n - 1;
        for (; (1 << log_stride) > ELEMENTARY_SIZE; --log_stride) {
          dim3 block(MAX_BLOCK_SIZE / 2);
          dim3 grid(div_up(batch_size, WORK_PER_THREAD), div_up(n / 2, MAX_BLOCK_SIZE / 2), nstack);
          butterfly_multiply_untied_backward_onestep_cuda_kernel<scalar_t, accscalar_t, true>
            <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, output_a, d_twiddle_a, d_input_a, log_stride, log_n);
        }
        int stride = 1 << log_stride;
        dim3 block(stride, div_up(MAX_BLOCK_SIZE, stride * 2));
        dim3 grid(div_up(batch_size, block.y), div_up(n / 2, stride), nstack);
        // dim3 block(stride);
        // dim3 grid(batch_size, div_up(n / 2, stride), nstack);
        butterfly_multiply_untied_backward_cuda_kernel<scalar_t, accscalar_t, true>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, output_a, d_twiddle_a, d_input_a, log_stride, log_n);
      } else {
        int stride = std::min<int>(ELEMENTARY_SIZE, n / 2);
        int log_stride = int(log2((double) stride));
        dim3 block(stride, div_up(MAX_BLOCK_SIZE, stride * 2));
        dim3 grid(div_up(batch_size, block.y), div_up(n / 2, stride), nstack);
        // dim3 block(stride);
        // dim3 grid(batch_size, div_up(n / 2, stride), nstack);
        butterfly_multiply_untied_backward_cuda_kernel<scalar_t, accscalar_t, false>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, output_a, d_twiddle_a, d_input_a, log_stride, log_n);
        for (log_stride++; log_stride <= log_n - 1; ++log_stride) {
          dim3 block(MAX_BLOCK_SIZE / 2);
          dim3 grid(div_up(batch_size, WORK_PER_THREAD), div_up(n / 2, MAX_BLOCK_SIZE / 2), nstack);
          butterfly_multiply_untied_backward_onestep_cuda_kernel<scalar_t, accscalar_t, false>
            <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, output_a, d_twiddle_a, d_input_a, log_stride, log_n);
        }
      }
    } else {  // complex
      const auto twiddle_a = twiddle.packed_accessor<scalar_t, 6>();
      const auto output_a = output.packed_accessor<scalar_t, 5>();
      auto d_twiddle_a = d_twiddle.packed_accessor<scalar_t, 6>();
      auto d_input_a = d_input.packed_accessor<scalar_t, 4>();
      if (increasing_stride) {
        int log_stride = log_n - 1;
        for (; (1 << log_stride) > ELEMENTARY_SIZE; --log_stride) {
          dim3 block(MAX_BLOCK_SIZE / 2);
          dim3 grid(div_up(batch_size, WORK_PER_THREAD), div_up(n / 2, MAX_BLOCK_SIZE / 2), nstack);
          butterfly_multiply_untied_backward_onestep_complex_cuda_kernel<scalar_t, accscalar_t, true>
            <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, output_a, d_twiddle_a, d_input_a, log_stride, log_n);
        }
        int stride = 1 << log_stride;
        dim3 block(stride);
        dim3 grid(batch_size, div_up(n / 2, stride), nstack);
        butterfly_multiply_untied_backward_complex_cuda_kernel<scalar_t, accscalar_t, true>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, output_a, d_twiddle_a, d_input_a, log_stride, log_n);
      } else {
        int stride = std::min<int>(ELEMENTARY_SIZE, n / 2);
        int log_stride = int(log2((double) stride));
        dim3 block(stride);
        dim3 grid(batch_size, div_up(n / 2, stride), nstack);
        butterfly_multiply_untied_backward_complex_cuda_kernel<scalar_t, accscalar_t, false>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, output_a, d_twiddle_a, d_input_a, log_stride, log_n);
        for (log_stride++; log_stride <= log_n - 1; ++log_stride) {
          dim3 block(MAX_BLOCK_SIZE / 2);
          dim3 grid(div_up(batch_size, WORK_PER_THREAD), div_up(n / 2, MAX_BLOCK_SIZE / 2), nstack);
          butterfly_multiply_untied_backward_onestep_complex_cuda_kernel<scalar_t, accscalar_t, false>
            <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, output_a, d_twiddle_a, d_input_a, log_stride, log_n);
        }
      }
    }
  });
  AT_CHECK(cudaGetLastError() == cudaSuccess,
     "butterfly_multiply_untied_backward_cuda failed with error code ",
     cudaGetLastError());
}

template <typename scalar_t, typename accscalar_t, bool increasing_stride, int log_max_stride,
          typename Function0, typename Function1, typename Function2>
__global__ void butterfly_multiply_untied_forward_backward_cuda_kernel(const CudaAcsr<scalar_t, 5> twiddle_a,
                                                                       Function0 load_input,
                                                                       Function1 load_grad,
                                                                       CudaAcsr<scalar_t, 5> d_twiddle_a,
                                                                       Function2 save_d_input,
                                                                       int batch_size) {
  const int s = blockIdx.y + gridDim.y * blockIdx.z;  // For conv2d butterfly as well
  const int max_stride = 1 << log_max_stride;
  const int input_base_idx = 0;
  __shared__ scalar_t s_input[ELEMENTARY_SIZE * 2];
  __shared__ scalar_t s_twiddle[ELEMENTARY_SIZE][2][2];
  // Forward pass to compute the intermediate values
  scalar_t input_val_storage[MAX_N_FACTORS][2];  // Storing inputs for backward pass
  load_input(s_input);
  int b = blockIdx.x * blockDim.y + threadIdx.y;
  int tid_x = threadIdx.x;
  int tid_y = threadIdx.y;
  #pragma unroll
  for (int idx = 0; idx <= log_max_stride; ++idx) {  // Let's not skip steps for now
    int log_stride = increasing_stride ? idx : log_max_stride - idx;
    int stride = 1 << log_stride;
    if (tid_y == 0) {
      s_twiddle[tid_x][0][0] = twiddle_a[s][log_stride][input_base_idx / 2 + tid_x][0][0];
      s_twiddle[tid_x][0][1] = twiddle_a[s][log_stride][input_base_idx / 2 + tid_x][0][1];
      s_twiddle[tid_x][1][0] = twiddle_a[s][log_stride][input_base_idx / 2 + tid_x][1][0];
      s_twiddle[tid_x][1][1] = twiddle_a[s][log_stride][input_base_idx / 2 + tid_x][1][1];
    }
    int low_order_bits = tid_x & (stride - 1);  // int low_order_bits = tid_x % stride;
    int pos_x = 2 * (tid_x - low_order_bits) + low_order_bits;
    int pos_y = tid_y * max_stride * 2;
    int pos = pos_x + pos_y;
    __syncthreads();
    const scalar_t twiddle_val[2][2] = {{s_twiddle[tid_x][0][0], s_twiddle[tid_x][0][1]},
                                        {s_twiddle[tid_x][1][0], s_twiddle[tid_x][1][1]}};
    if (b < batch_size) {
      const scalar_t input_val[2] = {s_input[pos], s_input[pos + stride]};
      input_val_storage[idx][0] = input_val[0];
      input_val_storage[idx][1] = input_val[1];
      s_input[pos] = twiddle_val[0][0] * input_val[0] + twiddle_val[0][1] * input_val[1];
      s_input[pos + stride] = twiddle_val[1][0] * input_val[0] + twiddle_val[1][1] * input_val[1];
    }
    __syncthreads();
    // otherwise some thread might go back to writing to s_twiddle before other thread can read
    // or s_s_input will be overwritten with s_grad before some thread can read
  }
  // Backward pass
  scalar_t* s_grad = &s_input[0]; // Reusing the same storage as s_input
  __shared__ accscalar_t s_d_twiddle[ELEMENTARY_SIZE][2][2];
  load_grad(s_grad);
  #pragma unroll
  for (int idx = log_max_stride; idx >= 0; --idx) {
    int log_stride = increasing_stride ? idx : log_max_stride - idx;
    int stride = 1 << log_stride;
    // tid_y == 0 is writing (atomicAdd) so tid_y == -1 can do the reading, instead of having to wait for tid_y == 0
    if (tid_y == blockDim.y - 1) {
      s_twiddle[tid_x][0][0] = twiddle_a[s][log_stride][input_base_idx / 2 + tid_x][0][0];
      s_twiddle[tid_x][0][1] = twiddle_a[s][log_stride][input_base_idx / 2 + tid_x][0][1];
      s_twiddle[tid_x][1][0] = twiddle_a[s][log_stride][input_base_idx / 2 + tid_x][1][0];
      s_twiddle[tid_x][1][1] = twiddle_a[s][log_stride][input_base_idx / 2 + tid_x][1][1];
    }
    int low_order_bits = tid_x & (stride - 1);  // int low_order_bits = tid_x % stride;
    int pos_x = 2 * (tid_x - low_order_bits) + low_order_bits;
    int pos_y = tid_y * max_stride * 2;
    int pos = pos_x + pos_y;
    __syncthreads();
    const scalar_t twiddle_val[2][2] = {{s_twiddle[tid_x][0][0], s_twiddle[tid_x][0][1]},
                                        {s_twiddle[tid_x][1][0], s_twiddle[tid_x][1][1]}};
    if (b < batch_size) {
      const scalar_t grad_val[2] = {s_grad[pos], s_grad[pos + stride]};
      s_grad[pos] = twiddle_val[0][0] * grad_val[0] + twiddle_val[1][0] * grad_val[1];
      s_grad[pos + stride] = twiddle_val[0][1] * grad_val[0] + twiddle_val[1][1] * grad_val[1];
      const scalar_t input_val[2] = {input_val_storage[idx][0], input_val_storage[idx][1]};
      s_d_twiddle[tid_x + tid_y * max_stride][0][0] = grad_val[0] * input_val[0];
      s_d_twiddle[tid_x + tid_y * max_stride][0][1] = grad_val[0] * input_val[1];
      s_d_twiddle[tid_x + tid_y * max_stride][1][0] = grad_val[1] * input_val[0];
      s_d_twiddle[tid_x + tid_y * max_stride][1][1] = grad_val[1] * input_val[1];
    }
    __syncthreads();
    if (tid_y == 0) {
      accscalar_t d_twiddle_val[2][2] = {{0, 0}, {0, 0}};
      for (int i = 0; i < blockDim.y; ++i) {
        if (blockIdx.x * blockDim.y + i < batch_size) {
          d_twiddle_val[0][0] += s_d_twiddle[tid_x + i * max_stride][0][0];
          d_twiddle_val[0][1] += s_d_twiddle[tid_x + i * max_stride][0][1];
          d_twiddle_val[1][0] += s_d_twiddle[tid_x + i * max_stride][1][0];
          d_twiddle_val[1][1] += s_d_twiddle[tid_x + i * max_stride][1][1];
        }
      }
      atomicAdd(&d_twiddle_a[s][log_stride][input_base_idx / 2 + tid_x][0][0], d_twiddle_val[0][0]);
      atomicAdd(&d_twiddle_a[s][log_stride][input_base_idx / 2 + tid_x][0][1], d_twiddle_val[0][1]);
      atomicAdd(&d_twiddle_a[s][log_stride][input_base_idx / 2 + tid_x][1][0], d_twiddle_val[1][0]);
      atomicAdd(&d_twiddle_a[s][log_stride][input_base_idx / 2 + tid_x][1][1], d_twiddle_val[1][1]);
    }
  }
  save_d_input(s_grad);
}

void butterfly_multiply_untied_forward_backward_cuda(const at::Tensor& twiddle, const at::Tensor& input, const at::Tensor& grad,
                                                     at::Tensor& d_twiddle, at::Tensor& d_input, bool increasing_stride) {
  int batch_size = input.size(0);
  const int nstack = input.size(1);
  const int n = input.size(2);
  const int log_n = int(log2((double) n));
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "butterfly_multiply_untied_forward_backward_cuda", [&] {
    using accscalar_t = at::acc_type<scalar_t, true>;
    const auto twiddle_a = twiddle.packed_accessor<scalar_t, 5, at::RestrictPtrTraits, int32_t>();
    const auto input_a = input.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>();
    const auto grad_a = grad.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>();
    auto d_twiddle_a = d_twiddle.packed_accessor<scalar_t, 5, at::RestrictPtrTraits, int32_t>();
    auto d_input_a = d_input.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>();
    int stride = std::min<int>(ELEMENTARY_SIZE, n / 2);
    int log_stride = int(log2((double) stride));
    dim3 block(stride, div_up(MAX_BLOCK_SIZE, stride * 2));
    dim3 grid(div_up(batch_size, block.y), 1, nstack);
    auto load_input = [batch_size, stride, input_a] __device__ (scalar_t* s_input) {
      const int b = blockIdx.x * blockDim.y + threadIdx.y;
      const int s = blockIdx.z;
      if (b < batch_size) {
        for (int i = threadIdx.x; i < stride * 2; i += blockDim.x) {
          s_input[i + threadIdx.y * stride * 2] = input_a[b][s][i];
        }
      }
    };
    auto load_grad = [batch_size, stride, grad_a] __device__ (scalar_t* s_grad) {
      const int b = blockIdx.x * blockDim.y + threadIdx.y;
      const int s = blockIdx.z;
      if (b < batch_size) {
        for (int i = threadIdx.x; i < stride * 2; i += blockDim.x) {
          s_grad[i + threadIdx.y * stride * 2] = grad_a[b][s][i];
        }
      }
    };
    auto save_d_input = [batch_size, stride, d_input_a] __device__ (scalar_t* s_grad) mutable {
      const int b = blockIdx.x * blockDim.y + threadIdx.y;
      const int s = blockIdx.z;
      if (b < batch_size) {
        for (int i = threadIdx.x; i < stride * 2; i += blockDim.x) {
          d_input_a[b][s][i] = s_grad[i + threadIdx.y * stride * 2];
        }
      }
    };
    switch (log_stride)
      {
      case 0:
        increasing_stride ? butterfly_multiply_untied_forward_backward_cuda_kernel<scalar_t, accscalar_t, true, 0>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, load_input, load_grad, d_twiddle_a, save_d_input, batch_size)
                          : butterfly_multiply_untied_forward_backward_cuda_kernel<scalar_t, accscalar_t, false, 0>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, load_input, load_grad, d_twiddle_a, save_d_input, batch_size); break;
      case 1:
        increasing_stride ? butterfly_multiply_untied_forward_backward_cuda_kernel<scalar_t, accscalar_t, true, 1>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, load_input, load_grad, d_twiddle_a, save_d_input, batch_size)
                          : butterfly_multiply_untied_forward_backward_cuda_kernel<scalar_t, accscalar_t, false, 1>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, load_input, load_grad, d_twiddle_a, save_d_input, batch_size); break;
      case 2:
        increasing_stride ? butterfly_multiply_untied_forward_backward_cuda_kernel<scalar_t, accscalar_t, true, 2>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, load_input, load_grad, d_twiddle_a, save_d_input, batch_size)
                          : butterfly_multiply_untied_forward_backward_cuda_kernel<scalar_t, accscalar_t, false, 2>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, load_input, load_grad, d_twiddle_a, save_d_input, batch_size); break;
      case 3:
        increasing_stride ? butterfly_multiply_untied_forward_backward_cuda_kernel<scalar_t, accscalar_t, true, 3>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, load_input, load_grad, d_twiddle_a, save_d_input, batch_size)
                          : butterfly_multiply_untied_forward_backward_cuda_kernel<scalar_t, accscalar_t, false, 3>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, load_input, load_grad, d_twiddle_a, save_d_input, batch_size); break;
      case 4:
        increasing_stride ? butterfly_multiply_untied_forward_backward_cuda_kernel<scalar_t, accscalar_t, true, 4>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, load_input, load_grad, d_twiddle_a, save_d_input, batch_size)
                          : butterfly_multiply_untied_forward_backward_cuda_kernel<scalar_t, accscalar_t, false, 4>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, load_input, load_grad, d_twiddle_a, save_d_input, batch_size); break;
      case 5:
        increasing_stride ? butterfly_multiply_untied_forward_backward_cuda_kernel<scalar_t, accscalar_t, true, 5>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, load_input, load_grad, d_twiddle_a, save_d_input, batch_size)
                          : butterfly_multiply_untied_forward_backward_cuda_kernel<scalar_t, accscalar_t, false, 5>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, load_input, load_grad, d_twiddle_a, save_d_input, batch_size); break;
      case 6:
        increasing_stride ? butterfly_multiply_untied_forward_backward_cuda_kernel<scalar_t, accscalar_t, true, 6>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, load_input, load_grad, d_twiddle_a, save_d_input, batch_size)
                          : butterfly_multiply_untied_forward_backward_cuda_kernel<scalar_t, accscalar_t, false, 6>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, load_input, load_grad, d_twiddle_a, save_d_input, batch_size); break;
      case 7:
        increasing_stride ? butterfly_multiply_untied_forward_backward_cuda_kernel<scalar_t, accscalar_t, true, 7>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, load_input, load_grad, d_twiddle_a, save_d_input, batch_size)
                          : butterfly_multiply_untied_forward_backward_cuda_kernel<scalar_t, accscalar_t, false, 7>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, load_input, load_grad, d_twiddle_a, save_d_input, batch_size); break;
      case 8:
        increasing_stride ? butterfly_multiply_untied_forward_backward_cuda_kernel<scalar_t, accscalar_t, true, 8>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, load_input, load_grad, d_twiddle_a, save_d_input, batch_size)
                          : butterfly_multiply_untied_forward_backward_cuda_kernel<scalar_t, accscalar_t, false, 8>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, load_input, load_grad, d_twiddle_a, save_d_input, batch_size); break;
      case 9:
        increasing_stride ? butterfly_multiply_untied_forward_backward_cuda_kernel<scalar_t, accscalar_t, true, 9>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, load_input, load_grad, d_twiddle_a, save_d_input, batch_size)
                          : butterfly_multiply_untied_forward_backward_cuda_kernel<scalar_t, accscalar_t, false, 9>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, load_input, load_grad, d_twiddle_a, save_d_input, batch_size); break;
      }
  });
  AT_CHECK(cudaGetLastError() == cudaSuccess,
     "butterfly_multiply_untied_forward_backward_cuda failed with error code ",
     cudaGetLastError());
}

template <typename scalar_t, bool increasing_stride, typename Function0, typename Function1>
__global__ void butterfly_ortho_multiply_tied_cuda_kernel(const CudaAcsr<scalar_t, 2> twiddle_cos_a,
                                                          const CudaAcsr<scalar_t, 2> twiddle_sin_a,
                                                          Function0 load_input,
                                                          Function1 save_output,
                                                          int log_max_stride,
                                                          int batch_size) {
  const int s = blockIdx.y + gridDim.y * blockIdx.z;  // For conv2d butterfly_ortho as well
  const int max_stride = 1 << log_max_stride;
  __shared__ scalar_t s_input[ELEMENTARY_SIZE * 2];
  __shared__ scalar_t s_twiddle[ELEMENTARY_SIZE][2];
  load_input(s_input);
  int b = blockIdx.x * blockDim.y + threadIdx.y;
  int tid_x = threadIdx.x;
  int tid_y = threadIdx.y;
  for (int idx = 0; idx < (log_max_stride + 1); ++idx) {
    int log_stride = increasing_stride ? idx : log_max_stride - idx;
    int stride = 1 << log_stride;
    int twiddle_start_idx = stride - 1;
    if ((tid_y == 0) && (tid_x < stride)) {
      s_twiddle[tid_x][0] = twiddle_cos_a[s][twiddle_start_idx + tid_x];
      s_twiddle[tid_x][1] = twiddle_sin_a[s][twiddle_start_idx + tid_x];
    }
    int low_order_bits = tid_x & (stride - 1);  // int low_order_bits = tid_x % stride;
    int twiddle_idx = low_order_bits;
    int pos_x = 2 * (tid_x - low_order_bits) + low_order_bits;
    int pos_y = tid_y * max_stride * 2;
    int pos = pos_x + pos_y;
    __syncthreads();
    const scalar_t twiddle_val[2] = {s_twiddle[twiddle_idx][0], s_twiddle[twiddle_idx][1]};
    if (b < batch_size) {
      const scalar_t input_val[2] = {s_input[pos], s_input[pos + stride]};
      s_input[pos] = twiddle_val[0] * input_val[0] - twiddle_val[1] * input_val[1];
      s_input[pos + stride] = twiddle_val[1] * input_val[0] + twiddle_val[0] * input_val[1];
    }
    __syncthreads();
    // otherwise some thread might go back to writing to s_twiddle before other thread can read
  }
  save_output(s_input);
}

void butterfly_ortho_multiply_tied_cuda(const at::Tensor& twiddle_cos, const at::Tensor& twiddle_sin, const at::Tensor& input, at::Tensor& output, bool increasing_stride) {
  int batch_size = input.size(0);
  const int nstack = input.size(1);
  const int n = input.size(2);
  const int log_n = int(log2((double) n));
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "butterfly_ortho_multiply_tied_cuda", [&] {
    using accscalar_t = at::acc_type<scalar_t, true>;
    const auto twiddle_cos_a = twiddle_cos.packed_accessor<scalar_t, 2, at::RestrictPtrTraits, int32_t>();
    const auto twiddle_sin_a = twiddle_sin.packed_accessor<scalar_t, 2, at::RestrictPtrTraits, int32_t>();
    const auto input_a = input.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>();
    auto output_a = output.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>();
    int stride = std::min<int>(ELEMENTARY_SIZE, n / 2);
    int log_stride = int(log2((double) stride));
    dim3 block(stride, div_up(MAX_BLOCK_SIZE, stride * 2));
    dim3 grid(div_up(batch_size, block.y), 1, nstack);
    auto load_input = [batch_size, stride, input_a] __device__ (scalar_t* s_input) {
      const int b = blockIdx.x * blockDim.y + threadIdx.y;
      const int s = blockIdx.z;
      if (b < batch_size) {
        for (int i = threadIdx.x; i < stride * 2; i += blockDim.x) {
          s_input[i + threadIdx.y * stride * 2] = input_a[b][s][i];
        }
      }
    };
    auto save_output = [batch_size, stride, output_a] __device__ (scalar_t* s_input) mutable {
      const int b = blockIdx.x * blockDim.y + threadIdx.y;
      const int s = blockIdx.z;
      if (b < batch_size) {
        for (int i = threadIdx.x; i < stride * 2; i += blockDim.x) {
          output_a[b][s][i] = s_input[i + threadIdx.y * stride * 2];
        }
      }
    };
    increasing_stride ? butterfly_ortho_multiply_tied_cuda_kernel<scalar_t, true>
      <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_cos_a, twiddle_sin_a, load_input, save_output, log_stride, batch_size)
                      : butterfly_ortho_multiply_tied_cuda_kernel<scalar_t, false>
      <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_cos_a, twiddle_sin_a, load_input, save_output, log_stride, batch_size);
  });
  AT_CHECK(cudaGetLastError() == cudaSuccess,
     "butterfly_ortho_multiply_tied_cuda failed with error code ",
     cudaGetLastError());
}

template <typename scalar_t, typename accscalar_t, bool increasing_stride,
          typename Function0, typename Function1, typename Function2>
__global__ void butterfly_ortho_multiply_tied_backward_cuda_kernel(const CudaAcsr<scalar_t, 2> twiddle_cos_a,
                                                                   const CudaAcsr<scalar_t, 2> twiddle_sin_a,
                                                                   Function0 load_output,
                                                                   Function1 load_grad,
                                                                   CudaAcsr<scalar_t, 2> d_twiddle_a,
                                                                   Function2 save_d_input,
                                                                   int log_max_stride,
                                                                   int batch_size) {
  const int s = blockIdx.y + gridDim.y * blockIdx.z;  // For conv2d butterfly_ortho as well
  const int max_stride = 1 << log_max_stride;
  __shared__ scalar_t s_output[ELEMENTARY_SIZE * 2];
  __shared__ scalar_t s_grad[ELEMENTARY_SIZE * 2];
  __shared__ scalar_t s_twiddle[ELEMENTARY_SIZE][2];
  __shared__ accscalar_t s_d_twiddle[ELEMENTARY_SIZE];
  int b = blockIdx.x * blockDim.y + threadIdx.y;
  int tid_x = threadIdx.x;
  int tid_y = threadIdx.y;
  load_output(s_output);
  load_grad(s_grad);
  for (int idx = log_max_stride; idx >= 0; --idx) {
    int log_stride = increasing_stride ? idx : log_max_stride - idx;
    int stride = 1 << log_stride;
    int twiddle_start_idx = stride - 1;
    // tid_y == 0 is writing (atomicAdd) so tid_y == -1 can do the reading, instead of having to wait for tid_y == 0
    if ((tid_y == blockDim.y - 1) && (tid_x < stride)) {
      s_twiddle[tid_x][0] = twiddle_cos_a[s][twiddle_start_idx + tid_x];
      s_twiddle[tid_x][1] = twiddle_sin_a[s][twiddle_start_idx + tid_x];
    }
    int low_order_bits = tid_x & (stride - 1);  // int low_order_bits = tid_x % stride;
    int twiddle_idx = low_order_bits;
    int pos_x = 2 * (tid_x - low_order_bits) + low_order_bits;
    int pos_y = tid_y * max_stride * 2;
    int pos = pos_x + pos_y;
    __syncthreads();
    const scalar_t twiddle_val[2] = {s_twiddle[twiddle_idx][0], s_twiddle[twiddle_idx][1]};
    scalar_t d_twiddle_val[1] = {0};  // Idk, to be consistent with sum_strided's interface
    if (b < batch_size) {
      const scalar_t grad_val[2] = {s_grad[pos], s_grad[pos + stride]};
      s_grad[pos] = twiddle_val[0] * grad_val[0] + twiddle_val[1] * grad_val[1];
      s_grad[pos + stride] = -twiddle_val[1] * grad_val[0] + twiddle_val[0] * grad_val[1];
      const scalar_t output_val[2] = {s_output[pos], s_output[pos + stride]};
      const scalar_t input_val[2] = {twiddle_val[0] * output_val[0] + twiddle_val[1] * output_val[1],
                                     -twiddle_val[1] * output_val[0] + twiddle_val[0] * output_val[1]};
      s_output[pos] = input_val[0];
      s_output[pos + stride] = input_val[1];
      d_twiddle_val[0]
        = (grad_val[0] * input_val[0] + grad_val[1] * input_val[1]) * (-twiddle_val[1])
        + (-grad_val[0] * input_val[1] + grad_val[1] * input_val[0]) * twiddle_val[0];
    }
    int tid = tid_x + tid_y * blockDim.x;
    int nthreads = blockDim.x * blockDim.y;
    sum_strided_atomic(reinterpret_cast<accscalar_t (&)[1]>(d_twiddle_val), s_d_twiddle, stride, nthreads, tid);
    if ((tid_y == 0) && (tid_x < stride)) {
      atomicAdd(&d_twiddle_a[s][twiddle_start_idx + twiddle_idx], s_d_twiddle[twiddle_idx]);
    }
  }
  save_d_input(s_grad);
}

void butterfly_ortho_multiply_tied_backward_cuda(const at::Tensor& twiddle_cos, const at::Tensor& twiddle_sin, const at::Tensor& output,
                                                 const at::Tensor& grad, at::Tensor& d_twiddle, at::Tensor& d_input, bool increasing_stride) {
  int batch_size = output.size(0);
  const int nstack = output.size(1);
  const int n = output.size(2);
  const int log_n = int(log2((double) n));
  AT_DISPATCH_FLOATING_TYPES(output.scalar_type(), "butterfly_ortho_multiply_tied_backward_cuda", [&] {
    using accscalar_t = at::acc_type<scalar_t, true>;
    const auto twiddle_cos_a = twiddle_cos.packed_accessor<scalar_t, 2, at::RestrictPtrTraits, int32_t>();
    const auto twiddle_sin_a = twiddle_sin.packed_accessor<scalar_t, 2, at::RestrictPtrTraits, int32_t>();
    const auto output_a = output.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>();
    const auto grad_a = grad.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>();
    auto d_twiddle_a = d_twiddle.packed_accessor<scalar_t, 2, at::RestrictPtrTraits, int32_t>();
    auto d_input_a = d_input.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>();
    int stride = std::min<int>(ELEMENTARY_SIZE, n / 2);
    int log_stride = int(log2((double) stride));
    dim3 block(stride, div_up(MAX_BLOCK_SIZE, stride * 2));
    dim3 grid(div_up(batch_size, block.y), 1, nstack);
    auto load_output = [batch_size, stride, output_a] __device__ (scalar_t* s_output) {
      const int b = blockIdx.x * blockDim.y + threadIdx.y;
      const int s = blockIdx.z;
      if (b < batch_size) {
        for (int i = threadIdx.x; i < stride * 2; i += blockDim.x) {
          s_output[i + threadIdx.y * stride * 2] = output_a[b][s][i];
        }
      }
    };
    auto load_grad = [batch_size, stride, grad_a] __device__ (scalar_t* s_grad) {
      const int b = blockIdx.x * blockDim.y + threadIdx.y;
      const int s = blockIdx.z;
      if (b < batch_size) {
        for (int i = threadIdx.x; i < stride * 2; i += blockDim.x) {
          s_grad[i + threadIdx.y * stride * 2] = grad_a[b][s][i];
        }
      }
    };
    auto save_d_input = [batch_size, stride, d_input_a] __device__ (scalar_t* s_grad) mutable {
      const int b = blockIdx.x * blockDim.y + threadIdx.y;
      const int s = blockIdx.z;
      if (b < batch_size) {
        for (int i = threadIdx.x; i < stride * 2; i += blockDim.x) {
          d_input_a[b][s][i] = s_grad[i + threadIdx.y * stride * 2];
        }
      }
    };
    increasing_stride ? butterfly_ortho_multiply_tied_backward_cuda_kernel<scalar_t, accscalar_t, true>
      <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_cos_a, twiddle_sin_a, load_output, load_grad, d_twiddle_a, save_d_input, log_stride, batch_size)
                      : butterfly_ortho_multiply_tied_backward_cuda_kernel<scalar_t, accscalar_t, false>
      <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_cos_a, twiddle_sin_a, load_output, load_grad, d_twiddle_a, save_d_input, log_stride, batch_size);
  });
  AT_CHECK(cudaGetLastError() == cudaSuccess,
     "butterfly_ortho_multiply_tied_backward_cuda failed with error code ",
     cudaGetLastError());
}


template <typename scalar_t, bool increasing_stride, typename Function0, typename Function1>
__global__ void butterfly_ortho_multiply_untied_cuda_kernel(const CudaAcsr<scalar_t, 3> twiddle_cos_a,
                                                            const CudaAcsr<scalar_t, 3> twiddle_sin_a,
                                                            Function0 load_input,
                                                            Function1 save_output,
                                                            int log_max_stride,
                                                            int batch_size) {
  const int s = blockIdx.y + gridDim.y * blockIdx.z;  // For conv2d butterfly_ortho as well
  const int max_stride = 1 << log_max_stride;
  const int input_base_idx = 0;
  __shared__ scalar_t s_input[ELEMENTARY_SIZE * 2];
  __shared__ scalar_t s_twiddle[ELEMENTARY_SIZE][2];
  load_input(s_input);
  int b = blockIdx.x * blockDim.y + threadIdx.y;
  int tid_x = threadIdx.x;
  int tid_y = threadIdx.y;
  for (int idx = 0; idx < (log_max_stride + 1); ++idx) {
    int log_stride = increasing_stride ? idx : log_max_stride - idx;
    int stride = 1 << log_stride;
    if (tid_y == 0) {
      s_twiddle[tid_x][0] = twiddle_cos_a[s][log_stride][input_base_idx / 2 + tid_x];
      s_twiddle[tid_x][1] = twiddle_sin_a[s][log_stride][input_base_idx / 2 + tid_x];
    }
    int low_order_bits = tid_x & (stride - 1);  // int low_order_bits = tid_x % stride;
    int pos_x = 2 * (tid_x - low_order_bits) + low_order_bits;
    int pos_y = tid_y * max_stride * 2;
    int pos = pos_x + pos_y;
    __syncthreads();
    const scalar_t twiddle_val[2] = {s_twiddle[tid_x][0], s_twiddle[tid_x][1]};
    if (b < batch_size) {
      const scalar_t input_val[2] = {s_input[pos], s_input[pos + stride]};
      s_input[pos] = twiddle_val[0] * input_val[0] - twiddle_val[1] * input_val[1];
      s_input[pos + stride] = twiddle_val[1] * input_val[0] + twiddle_val[0] * input_val[1];
    }
    __syncthreads();
    // otherwise some thread might go back to writing to s_twiddle before other thread can read
  }
  save_output(s_input);
}

void butterfly_ortho_multiply_untied_cuda(const at::Tensor& twiddle_cos, const at::Tensor& twiddle_sin, const at::Tensor& input, at::Tensor& output, bool increasing_stride) {
  int batch_size = input.size(0);
  const int nstack = input.size(1);
  const int n = input.size(2);
  const int log_n = int(log2((double) n));
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "butterfly_ortho_multiply_untied_cuda", [&] {
    using accscalar_t = at::acc_type<scalar_t, true>;
    const auto twiddle_cos_a = twiddle_cos.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>();
    const auto twiddle_sin_a = twiddle_sin.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>();
    const auto input_a = input.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>();
    auto output_a = output.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>();
    int stride = std::min<int>(ELEMENTARY_SIZE, n / 2);
    int log_stride = int(log2((double) stride));
    dim3 block(stride, div_up(MAX_BLOCK_SIZE, stride * 2));
    dim3 grid(div_up(batch_size, block.y), 1, nstack);
    auto load_input = [batch_size, stride, input_a] __device__ (scalar_t* s_input) {
      const int b = blockIdx.x * blockDim.y + threadIdx.y;
      const int s = blockIdx.z;
      if (b < batch_size) {
        for (int i = threadIdx.x; i < stride * 2; i += blockDim.x) {
          s_input[i + threadIdx.y * stride * 2] = input_a[b][s][i];
        }
      }
    };
    auto save_output = [batch_size, stride, output_a] __device__ (scalar_t* s_input) mutable {
      const int b = blockIdx.x * blockDim.y + threadIdx.y;
      const int s = blockIdx.z;
      if (b < batch_size) {
        for (int i = threadIdx.x; i < stride * 2; i += blockDim.x) {
          output_a[b][s][i] = s_input[i + threadIdx.y * stride * 2];
        }
      }
    };
    increasing_stride ? butterfly_ortho_multiply_untied_cuda_kernel<scalar_t, true>
      <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_cos_a, twiddle_sin_a, load_input, save_output, log_stride, batch_size)
                      : butterfly_ortho_multiply_untied_cuda_kernel<scalar_t, false>
      <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_cos_a, twiddle_sin_a, load_input, save_output, log_stride, batch_size);
  });
  AT_CHECK(cudaGetLastError() == cudaSuccess,
     "butterfly_ortho_multiply_untied_cuda failed with error code ",
     cudaGetLastError());
}

template <typename scalar_t, typename accscalar_t, bool increasing_stride,
          typename Function0, typename Function1, typename Function2>
__global__ void butterfly_ortho_multiply_untied_backward_cuda_kernel(const CudaAcsr<scalar_t, 3> twiddle_cos_a,
                                                                     const CudaAcsr<scalar_t, 3> twiddle_sin_a,
                                                                     Function0 load_output,
                                                                     Function1 load_grad,
                                                                     CudaAcsr<scalar_t, 3> d_twiddle_a,
                                                                     Function2 save_d_input,
                                                                     int log_max_stride,
                                                                     int batch_size) {
  const int s = blockIdx.y + gridDim.y * blockIdx.z;  // For conv2d butterfly_ortho as well
  const int max_stride = 1 << log_max_stride;
  const int input_base_idx = 0;
  __shared__ scalar_t s_output[ELEMENTARY_SIZE * 2];
  __shared__ scalar_t s_grad[ELEMENTARY_SIZE * 2];
  __shared__ scalar_t s_twiddle[ELEMENTARY_SIZE][2];
  __shared__ accscalar_t s_d_twiddle[ELEMENTARY_SIZE];
  int b = blockIdx.x * blockDim.y + threadIdx.y;
  int tid_x = threadIdx.x;
  int tid_y = threadIdx.y;
  load_output(s_output);
  load_grad(s_grad);
  for (int idx = log_max_stride; idx >= 0; --idx) {
    int log_stride = increasing_stride ? idx : log_max_stride - idx;
    int stride = 1 << log_stride;
    // tid_y == 0 is writing (atomicAdd) so tid_y == -1 can do the reading, instead of having to wait for tid_y == 0
    if (tid_y == blockDim.y - 1) {
      s_twiddle[tid_x][0] = twiddle_cos_a[s][log_stride][input_base_idx / 2 + tid_x];
      s_twiddle[tid_x][1] = twiddle_sin_a[s][log_stride][input_base_idx / 2 + tid_x];
    }
    int low_order_bits = tid_x & (stride - 1);  // int low_order_bits = tid_x % stride;
    int pos_x = 2 * (tid_x - low_order_bits) + low_order_bits;
    int pos_y = tid_y * max_stride * 2;
    int pos = pos_x + pos_y;
    __syncthreads();
    const scalar_t twiddle_val[2] = {s_twiddle[tid_x][0], s_twiddle[tid_x][1]};
    if (b < batch_size) {
      const scalar_t grad_val[2] = {s_grad[pos], s_grad[pos + stride]};
      s_grad[pos] = twiddle_val[0] * grad_val[0] + twiddle_val[1] * grad_val[1];
      s_grad[pos + stride] = -twiddle_val[1] * grad_val[0] + twiddle_val[0] * grad_val[1];
      const scalar_t output_val[2] = {s_output[pos], s_output[pos + stride]};
      const scalar_t input_val[2] = {twiddle_val[0] * output_val[0] + twiddle_val[1] * output_val[1],
                                     -twiddle_val[1] * output_val[0] + twiddle_val[0] * output_val[1]};
      s_output[pos] = input_val[0];
      s_output[pos + stride] = input_val[1];
      s_d_twiddle[tid_x + tid_y * max_stride]
        = (grad_val[0] * input_val[0] + grad_val[1] * input_val[1]) * (-twiddle_val[1])
        + (-grad_val[0] * input_val[1] + grad_val[1] * input_val[0]) * twiddle_val[0];
    }
    __syncthreads();
    if (tid_y == 0) {
      accscalar_t d_twiddle_val = 0;
      for (int i = 0; i < blockDim.y; ++i) {
        if (blockIdx.x * blockDim.y + i < batch_size) {
          d_twiddle_val += s_d_twiddle[tid_x + i * max_stride];
        }
      }
      atomicAdd(&d_twiddle_a[s][log_stride][input_base_idx / 2 + tid_x], d_twiddle_val);
    }
  }
  save_d_input(s_grad);
}

void butterfly_ortho_multiply_untied_backward_cuda(const at::Tensor& twiddle_cos, const at::Tensor& twiddle_sin, const at::Tensor& output,
                                                   const at::Tensor& grad, at::Tensor& d_twiddle, at::Tensor& d_input, bool increasing_stride) {
  int batch_size = output.size(0);
  const int nstack = output.size(1);
  const int n = output.size(2);
  const int log_n = int(log2((double) n));
  AT_DISPATCH_FLOATING_TYPES(output.scalar_type(), "butterfly_ortho_multiply_untied_backward_cuda", [&] {
    using accscalar_t = at::acc_type<scalar_t, true>;
    const auto twiddle_cos_a = twiddle_cos.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>();
    const auto twiddle_sin_a = twiddle_sin.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>();
    const auto output_a = output.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>();
    const auto grad_a = grad.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>();
    auto d_twiddle_a = d_twiddle.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>();
    auto d_input_a = d_input.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>();
    int stride = std::min<int>(ELEMENTARY_SIZE, n / 2);
    int log_stride = int(log2((double) stride));
    dim3 block(stride, div_up(MAX_BLOCK_SIZE, stride * 2));
    dim3 grid(div_up(batch_size, block.y), 1, nstack);
    auto load_output = [batch_size, stride, output_a] __device__ (scalar_t* s_output) {
      const int b = blockIdx.x * blockDim.y + threadIdx.y;
      const int s = blockIdx.z;
      if (b < batch_size) {
        for (int i = threadIdx.x; i < stride * 2; i += blockDim.x) {
          s_output[i + threadIdx.y * stride * 2] = output_a[b][s][i];
        }
      }
    };
    auto load_grad = [batch_size, stride, grad_a] __device__ (scalar_t* s_grad) {
      const int b = blockIdx.x * blockDim.y + threadIdx.y;
      const int s = blockIdx.z;
      if (b < batch_size) {
        for (int i = threadIdx.x; i < stride * 2; i += blockDim.x) {
          s_grad[i + threadIdx.y * stride * 2] = grad_a[b][s][i];
        }
      }
    };
    auto save_d_input = [batch_size, stride, d_input_a] __device__ (scalar_t* s_grad) mutable {
      const int b = blockIdx.x * blockDim.y + threadIdx.y;
      const int s = blockIdx.z;
      if (b < batch_size) {
        for (int i = threadIdx.x; i < stride * 2; i += blockDim.x) {
          d_input_a[b][s][i] = s_grad[i + threadIdx.y * stride * 2];
        }
      }
    };
    increasing_stride ? butterfly_ortho_multiply_untied_backward_cuda_kernel<scalar_t, accscalar_t, true>
      <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_cos_a, twiddle_sin_a, load_output, load_grad, d_twiddle_a, save_d_input, log_stride, batch_size)
                      : butterfly_ortho_multiply_untied_backward_cuda_kernel<scalar_t, accscalar_t, false>
      <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_cos_a, twiddle_sin_a, load_output, load_grad, d_twiddle_a, save_d_input, log_stride, batch_size);
  });
  AT_CHECK(cudaGetLastError() == cudaSuccess,
     "butterfly_ortho_multiply_untied_backward_cuda failed with error code ",
     cudaGetLastError());
}

template <typename scalar_t, typename Function0, typename Function1>
__global__ void bbt_multiply_untied_cuda_kernel(const CudaAcsr<scalar_t, 5> twiddle_a,
                                                Function0 load_input,
                                                Function1 save_output,
                                                int log_max_stride,
                                                int batch_size,
                                                int nblocks) {
  const int s = blockIdx.y + gridDim.y * blockIdx.z;  // For conv2d bbt as well
  const int max_stride = 1 << log_max_stride;
  const int input_base_idx = 0;
  __shared__ scalar_t s_input[ELEMENTARY_SIZE * 2];
  __shared__ scalar_t s_twiddle[ELEMENTARY_SIZE][2][2];
  load_input(s_input);
  int b = blockIdx.x * blockDim.y + threadIdx.y;
  int tid_x = threadIdx.x;
  int tid_y = threadIdx.y;
  for (int block = 0; block < nblocks; ++block) {
    for (int idx = 0; idx < 2 * (log_max_stride + 1); ++idx) {
      int log_stride = idx <= log_max_stride ? log_max_stride - idx : idx - log_max_stride - 1;
      int stride = 1 << log_stride;
      if (tid_y == 0) {
        s_twiddle[tid_x][0][0] = twiddle_a[s][idx + block * 2 * (log_max_stride + 1)][input_base_idx / 2 + tid_x][0][0];
        s_twiddle[tid_x][0][1] = twiddle_a[s][idx + block * 2 * (log_max_stride + 1)][input_base_idx / 2 + tid_x][0][1];
        s_twiddle[tid_x][1][0] = twiddle_a[s][idx + block * 2 * (log_max_stride + 1)][input_base_idx / 2 + tid_x][1][0];
        s_twiddle[tid_x][1][1] = twiddle_a[s][idx + block * 2 * (log_max_stride + 1)][input_base_idx / 2 + tid_x][1][1];
      }
      int low_order_bits = tid_x & (stride - 1);  // int low_order_bits = tid_x % stride;
      int pos_x = 2 * (tid_x - low_order_bits) + low_order_bits;
      int pos_y = tid_y * max_stride * 2;
      int pos = pos_x + pos_y;
      __syncthreads();
      const scalar_t twiddle_val[2][2] = {{s_twiddle[tid_x][0][0], s_twiddle[tid_x][0][1]},
                                          {s_twiddle[tid_x][1][0], s_twiddle[tid_x][1][1]}};
      if (b < batch_size) {
        const scalar_t input_val[2] = {s_input[pos], s_input[pos + stride]};
        s_input[pos] = twiddle_val[0][0] * input_val[0] + twiddle_val[0][1] * input_val[1];
        s_input[pos + stride] = twiddle_val[1][0] * input_val[0] + twiddle_val[1][1] * input_val[1];
      }
      __syncthreads();
      // otherwise some thread might go back to writing to s_twiddle before other thread can read
    }
  }
  save_output(s_input);
}

void bbt_multiply_untied_cuda(const at::Tensor& twiddle, const at::Tensor& input, at::Tensor& output) {
  int batch_size = input.size(0);
  const int nstack = input.size(1);
  const int n = input.size(2);
  const int log_n = int(log2((double) n));
  int nblocks = twiddle.size(1) / (2 * log_n);
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "bbt_multiply_untied_cuda", [&] {
    using accscalar_t = at::acc_type<scalar_t, true>;
    const auto twiddle_a = twiddle.packed_accessor<scalar_t, 5, at::RestrictPtrTraits, int32_t>();
    const auto input_a = input.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>();
    auto output_a = output.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>();
    int stride = std::min<int>(ELEMENTARY_SIZE, n / 2);
    int log_stride = int(log2((double) stride));
    dim3 block(stride, div_up(MAX_BLOCK_SIZE, stride * 2));
    dim3 grid(div_up(batch_size, block.y), 1, nstack);
    auto load_input = [batch_size, stride, input_a] __device__ (scalar_t* s_input) {
      const int b = blockIdx.x * blockDim.y + threadIdx.y;
      const int s = blockIdx.z;
      if (b < batch_size) {
        for (int i = threadIdx.x; i < stride * 2; i += blockDim.x) {
          s_input[i + threadIdx.y * stride * 2] = input_a[b][s][i];
        }
      }
    };
    auto save_output = [batch_size, stride, output_a] __device__ (scalar_t* s_input) mutable {
      const int b = blockIdx.x * blockDim.y + threadIdx.y;
      const int s = blockIdx.z;
      if (b < batch_size) {
        for (int i = threadIdx.x; i < stride * 2; i += blockDim.x) {
          output_a[b][s][i] = s_input[i + threadIdx.y * stride * 2];
        }
      }
    };
    bbt_multiply_untied_cuda_kernel<scalar_t>
      <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, load_input, save_output, log_stride, batch_size, nblocks);
  });
  AT_CHECK(cudaGetLastError() == cudaSuccess,
     "bbt_multiply_untied_cuda failed with error code ",
     cudaGetLastError());
}

template <typename scalar_t, typename accscalar_t, int nblocks,
          typename Function0, typename Function1, typename Function2>
__global__ void bbt_multiply_untied_forward_backward_cuda_kernel(const CudaAcsr<scalar_t, 5> twiddle_a,
                                                                 Function0 load_input,
                                                                 Function1 load_grad,
                                                                 CudaAcsr<scalar_t, 5> d_twiddle_a,
                                                                 Function2 save_d_input,
                                                                 int log_max_stride,
                                                                 int batch_size) {
  const int s = blockIdx.y + gridDim.y * blockIdx.z;  // For conv2d bbt as well
  const int max_stride = 1 << log_max_stride;
  const int input_base_idx = 0;
  __shared__ scalar_t s_input[ELEMENTARY_SIZE * 2];
  __shared__ scalar_t s_twiddle[ELEMENTARY_SIZE][2][2];
  // Forward pass to compute the intermediate values
  scalar_t input_val_storage[nblocks * 2 * MAX_N_FACTORS][2];  // Storing inputs for backward pass
  load_input(s_input);
  int b = blockIdx.x * blockDim.y + threadIdx.y;
  int tid_x = threadIdx.x;
  int tid_y = threadIdx.y;
  for (int block = 0; block < nblocks; ++block) {
    for (int idx = 0; idx < 2 * (log_max_stride + 1); ++idx) {  // Let's not skip steps for now
      int log_stride = idx <= log_max_stride ? log_max_stride - idx : idx - log_max_stride - 1;
      int stride = 1 << log_stride;
      if (tid_y == 0) {
        s_twiddle[tid_x][0][0] = twiddle_a[s][idx + block * 2 * (log_max_stride + 1)][input_base_idx / 2 + tid_x][0][0];
        s_twiddle[tid_x][0][1] = twiddle_a[s][idx + block * 2 * (log_max_stride + 1)][input_base_idx / 2 + tid_x][0][1];
        s_twiddle[tid_x][1][0] = twiddle_a[s][idx + block * 2 * (log_max_stride + 1)][input_base_idx / 2 + tid_x][1][0];
        s_twiddle[tid_x][1][1] = twiddle_a[s][idx + block * 2 * (log_max_stride + 1)][input_base_idx / 2 + tid_x][1][1];
      }
      int low_order_bits = tid_x & (stride - 1);  // int low_order_bits = tid_x % stride;
      int pos_x = 2 * (tid_x - low_order_bits) + low_order_bits;
      int pos_y = tid_y * max_stride * 2;
      int pos = pos_x + pos_y;
      __syncthreads();
      const scalar_t twiddle_val[2][2] = {{s_twiddle[tid_x][0][0], s_twiddle[tid_x][0][1]},
                                          {s_twiddle[tid_x][1][0], s_twiddle[tid_x][1][1]}};
      if (b < batch_size) {
        const scalar_t input_val[2] = {s_input[pos], s_input[pos + stride]};
        input_val_storage[idx + block * 2 * (log_max_stride + 1)][0] = input_val[0];
        input_val_storage[idx + block * 2 * (log_max_stride + 1)][1] = input_val[1];
        s_input[pos] = twiddle_val[0][0] * input_val[0] + twiddle_val[0][1] * input_val[1];
        s_input[pos + stride] = twiddle_val[1][0] * input_val[0] + twiddle_val[1][1] * input_val[1];
      }
      __syncthreads();
      // otherwise some thread might go back to writing to s_twiddle before other thread can read
      // or s_s_input will be overwritten with s_grad before some thread can read
    }
  }
  // Backward pass
  scalar_t* s_grad = &s_input[0]; // Reusing the same storage as s_input
  __shared__ accscalar_t s_d_twiddle[ELEMENTARY_SIZE][2][2];
  load_grad(s_grad);
  for (int block = nblocks - 1; block >= 0; --block) {
    for (int idx = 2 * (log_max_stride + 1) - 1; idx >= 0; --idx) {
      int log_stride = idx <= log_max_stride ? log_max_stride - idx : idx - log_max_stride - 1;
      int stride = 1 << log_stride;
      // tid_y == 0 is writing (atomicAdd) so tid_y == -1 can do the reading, instead of having to wait for tid_y == 0
      if (tid_y == blockDim.y - 1) {
        s_twiddle[tid_x][0][0] = twiddle_a[s][idx + block * 2 * (log_max_stride + 1)][input_base_idx / 2 + tid_x][0][0];
        s_twiddle[tid_x][0][1] = twiddle_a[s][idx + block * 2 * (log_max_stride + 1)][input_base_idx / 2 + tid_x][0][1];
        s_twiddle[tid_x][1][0] = twiddle_a[s][idx + block * 2 * (log_max_stride + 1)][input_base_idx / 2 + tid_x][1][0];
        s_twiddle[tid_x][1][1] = twiddle_a[s][idx + block * 2 * (log_max_stride + 1)][input_base_idx / 2 + tid_x][1][1];
      }
      int low_order_bits = tid_x & (stride - 1);  // int low_order_bits = tid_x % stride;
      int pos_x = 2 * (tid_x - low_order_bits) + low_order_bits;
      int pos_y = tid_y * max_stride * 2;
      int pos = pos_x + pos_y;
      __syncthreads();
      const scalar_t twiddle_val[2][2] = {{s_twiddle[tid_x][0][0], s_twiddle[tid_x][0][1]},
                                          {s_twiddle[tid_x][1][0], s_twiddle[tid_x][1][1]}};
      if (b < batch_size) {
        const scalar_t grad_val[2] = {s_grad[pos], s_grad[pos + stride]};
        s_grad[pos] = twiddle_val[0][0] * grad_val[0] + twiddle_val[1][0] * grad_val[1];
        s_grad[pos + stride] = twiddle_val[0][1] * grad_val[0] + twiddle_val[1][1] * grad_val[1];
        const scalar_t input_val[2] = {input_val_storage[idx + block * 2 * (log_max_stride + 1)][0], input_val_storage[idx + block * 2 * (log_max_stride + 1)][1]};
        s_d_twiddle[tid_x + tid_y * max_stride][0][0] = grad_val[0] * input_val[0];
        s_d_twiddle[tid_x + tid_y * max_stride][0][1] = grad_val[0] * input_val[1];
        s_d_twiddle[tid_x + tid_y * max_stride][1][0] = grad_val[1] * input_val[0];
        s_d_twiddle[tid_x + tid_y * max_stride][1][1] = grad_val[1] * input_val[1];
      }
      __syncthreads();
      if (tid_y == 0) {
        accscalar_t d_twiddle_val[2][2] = {{0, 0}, {0, 0}};
        for (int i = 0; i < blockDim.y; ++i) {
          if (blockIdx.x * blockDim.y + i < batch_size) {
            d_twiddle_val[0][0] += s_d_twiddle[tid_x + i * max_stride][0][0];
            d_twiddle_val[0][1] += s_d_twiddle[tid_x + i * max_stride][0][1];
            d_twiddle_val[1][0] += s_d_twiddle[tid_x + i * max_stride][1][0];
            d_twiddle_val[1][1] += s_d_twiddle[tid_x + i * max_stride][1][1];
          }
        }
        atomicAdd(&d_twiddle_a[s][idx + block * 2 * (log_max_stride + 1)][input_base_idx / 2 + tid_x][0][0], d_twiddle_val[0][0]);
        atomicAdd(&d_twiddle_a[s][idx + block * 2 * (log_max_stride + 1)][input_base_idx / 2 + tid_x][0][1], d_twiddle_val[0][1]);
        atomicAdd(&d_twiddle_a[s][idx + block * 2 * (log_max_stride + 1)][input_base_idx / 2 + tid_x][1][0], d_twiddle_val[1][0]);
        atomicAdd(&d_twiddle_a[s][idx + block * 2 * (log_max_stride + 1)][input_base_idx / 2 + tid_x][1][1], d_twiddle_val[1][1]);
      }
    }
  }
  save_d_input(s_grad);
}

void bbt_multiply_untied_forward_backward_cuda(const at::Tensor& twiddle, const at::Tensor& input, const at::Tensor& grad,
                                               at::Tensor& d_twiddle, at::Tensor& d_input) {
  int batch_size = input.size(0);
  const int nstack = input.size(1);
  const int n = input.size(2);
  const int log_n = int(log2((double) n));
  int nblocks = twiddle.size(1) / (2 * log_n);
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "bbt_multiply_untied_forward_backward_cuda", [&] {
    using accscalar_t = at::acc_type<scalar_t, true>;
    const auto twiddle_a = twiddle.packed_accessor<scalar_t, 5, at::RestrictPtrTraits, int32_t>();
    const auto input_a = input.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>();
    const auto grad_a = grad.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>();
    auto d_twiddle_a = d_twiddle.packed_accessor<scalar_t, 5, at::RestrictPtrTraits, int32_t>();
    auto d_input_a = d_input.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>();
    int stride = std::min<int>(ELEMENTARY_SIZE, n / 2);
    int log_stride = int(log2((double) stride));
    dim3 block(stride, div_up(MAX_BLOCK_SIZE, stride * 2));
    dim3 grid(div_up(batch_size, block.y), 1, nstack);
    auto load_input = [batch_size, stride, input_a] __device__ (scalar_t* s_input) {
      const int b = blockIdx.x * blockDim.y + threadIdx.y;
      const int s = blockIdx.z;
      if (b < batch_size) {
        for (int i = threadIdx.x; i < stride * 2; i += blockDim.x) {
          s_input[i + threadIdx.y * stride * 2] = input_a[b][s][i];
        }
      }
    };
    auto load_grad = [batch_size, stride, grad_a] __device__ (scalar_t* s_grad) {
      const int b = blockIdx.x * blockDim.y + threadIdx.y;
      const int s = blockIdx.z;
      if (b < batch_size) {
        for (int i = threadIdx.x; i < stride * 2; i += blockDim.x) {
          s_grad[i + threadIdx.y * stride * 2] = grad_a[b][s][i];
        }
      }
    };
    auto save_d_input = [batch_size, stride, d_input_a] __device__ (scalar_t* s_grad) mutable {
      const int b = blockIdx.x * blockDim.y + threadIdx.y;
      const int s = blockIdx.z;
      if (b < batch_size) {
        for (int i = threadIdx.x; i < stride * 2; i += blockDim.x) {
          d_input_a[b][s][i] = s_grad[i + threadIdx.y * stride * 2];
        }
      }
    };
    switch (nblocks)
      {
      case 1:
        bbt_multiply_untied_forward_backward_cuda_kernel<scalar_t, accscalar_t, 1>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, load_input, load_grad, d_twiddle_a, save_d_input, log_stride, batch_size); break;
      case 2:
        bbt_multiply_untied_forward_backward_cuda_kernel<scalar_t, accscalar_t, 2>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, load_input, load_grad, d_twiddle_a, save_d_input, log_stride, batch_size); break;
      case 3:
        bbt_multiply_untied_forward_backward_cuda_kernel<scalar_t, accscalar_t, 3>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, load_input, load_grad, d_twiddle_a, save_d_input, log_stride, batch_size); break;
      case 4:
        bbt_multiply_untied_forward_backward_cuda_kernel<scalar_t, accscalar_t, 4>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, load_input, load_grad, d_twiddle_a, save_d_input, log_stride, batch_size); break;
      case 5:
        bbt_multiply_untied_forward_backward_cuda_kernel<scalar_t, accscalar_t, 5>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, load_input, load_grad, d_twiddle_a, save_d_input, log_stride, batch_size); break;
      case 6:
        bbt_multiply_untied_forward_backward_cuda_kernel<scalar_t, accscalar_t, 6>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, load_input, load_grad, d_twiddle_a, save_d_input, log_stride, batch_size); break;
      case 7:
        bbt_multiply_untied_forward_backward_cuda_kernel<scalar_t, accscalar_t, 7>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, load_input, load_grad, d_twiddle_a, save_d_input, log_stride, batch_size); break;
      case 8:
        bbt_multiply_untied_forward_backward_cuda_kernel<scalar_t, accscalar_t, 8>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, load_input, load_grad, d_twiddle_a, save_d_input, log_stride, batch_size); break;
      case 9:
        bbt_multiply_untied_forward_backward_cuda_kernel<scalar_t, accscalar_t, 9>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, load_input, load_grad, d_twiddle_a, save_d_input, log_stride, batch_size); break;
      case 10:
        bbt_multiply_untied_forward_backward_cuda_kernel<scalar_t, accscalar_t, 10>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, load_input, load_grad, d_twiddle_a, save_d_input, log_stride, batch_size); break;
      case 11:
        bbt_multiply_untied_forward_backward_cuda_kernel<scalar_t, accscalar_t, 11>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, load_input, load_grad, d_twiddle_a, save_d_input, log_stride, batch_size); break;
      case 12:
        bbt_multiply_untied_forward_backward_cuda_kernel<scalar_t, accscalar_t, 12>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, load_input, load_grad, d_twiddle_a, save_d_input, log_stride, batch_size); break;
      case 13:
        bbt_multiply_untied_forward_backward_cuda_kernel<scalar_t, accscalar_t, 13>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, load_input, load_grad, d_twiddle_a, save_d_input, log_stride, batch_size); break;
      case 14:
        bbt_multiply_untied_forward_backward_cuda_kernel<scalar_t, accscalar_t, 14>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, load_input, load_grad, d_twiddle_a, save_d_input, log_stride, batch_size); break;
      }
  });
  AT_CHECK(cudaGetLastError() == cudaSuccess,
     "bbt_multiply_untied_forward_backward_cuda failed with error code ",
     cudaGetLastError());
}

template <typename scalar_t, typename Function0, typename Function1>
__global__ void bbt_ortho_multiply_untied_cuda_kernel(const CudaAcsr<scalar_t, 3> twiddle_cos_a,
                                                      const CudaAcsr<scalar_t, 3> twiddle_sin_a,
                                                      Function0 load_input,
                                                      Function1 save_output,
                                                      int log_max_stride,
                                                      int batch_size,
                                                      int nblocks) {
  const int s = blockIdx.y + gridDim.y * blockIdx.z;  // For conv2d bbt_ortho as well
  const int max_stride = 1 << log_max_stride;
  const int input_base_idx = 0;
  __shared__ scalar_t s_input[ELEMENTARY_SIZE * 2];
  __shared__ scalar_t s_twiddle[ELEMENTARY_SIZE][2];
  load_input(s_input);
  int b = blockIdx.x * blockDim.y + threadIdx.y;
  int tid_x = threadIdx.x;
  int tid_y = threadIdx.y;
  for (int block = 0; block < nblocks; ++block) {
    for (int idx = 0; idx < 2 * (log_max_stride + 1); ++idx) {
      int log_stride = idx <= log_max_stride ? log_max_stride - idx : idx - log_max_stride - 1;
      int stride = 1 << log_stride;
      if (tid_y == 0) {
        s_twiddle[tid_x][0] = twiddle_cos_a[s][idx + block * 2 * (log_max_stride + 1)][input_base_idx / 2 + tid_x];
        s_twiddle[tid_x][1] = twiddle_sin_a[s][idx + block * 2 * (log_max_stride + 1)][input_base_idx / 2 + tid_x];
      }
      int low_order_bits = tid_x & (stride - 1);  // int low_order_bits = tid_x % stride;
      int pos_x = 2 * (tid_x - low_order_bits) + low_order_bits;
      int pos_y = tid_y * max_stride * 2;
      int pos = pos_x + pos_y;
      __syncthreads();
      const scalar_t twiddle_val[2] = {s_twiddle[tid_x][0], s_twiddle[tid_x][1]};
      if (b < batch_size) {
        const scalar_t input_val[2] = {s_input[pos], s_input[pos + stride]};
        s_input[pos] = twiddle_val[0] * input_val[0] - twiddle_val[1] * input_val[1];
        s_input[pos + stride] = twiddle_val[1] * input_val[0] + twiddle_val[0] * input_val[1];
      }
      __syncthreads();
      // otherwise some thread might go back to writing to s_twiddle before other thread can read
    }
  }
  save_output(s_input);
}

void bbt_ortho_multiply_untied_cuda(const at::Tensor& twiddle_cos, const at::Tensor& twiddle_sin,
                                    const at::Tensor& input, at::Tensor& output) {
  int batch_size = input.size(0);
  const int nstack = input.size(1);
  const int n = input.size(2);
  const int log_n = int(log2((double) n));
  int nblocks = twiddle_cos.size(1) / (2 * log_n);
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "bbt_ortho_multiply_untied_cuda", [&] {
    using accscalar_t = at::acc_type<scalar_t, true>;
    const auto twiddle_cos_a = twiddle_cos.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>();
    const auto twiddle_sin_a = twiddle_sin.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>();
    const auto input_a = input.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>();
    auto output_a = output.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>();
    int stride = std::min<int>(ELEMENTARY_SIZE, n / 2);
    int log_stride = int(log2((double) stride));
    dim3 block(stride, div_up(MAX_BLOCK_SIZE, stride * 2));
    dim3 grid(div_up(batch_size, block.y), 1, nstack);
    auto load_input = [batch_size, stride, input_a] __device__ (scalar_t* s_input) {
      const int b = blockIdx.x * blockDim.y + threadIdx.y;
      const int s = blockIdx.z;
      if (b < batch_size) {
        for (int i = threadIdx.x; i < stride * 2; i += blockDim.x) {
          s_input[i + threadIdx.y * stride * 2] = input_a[b][s][i];
        }
      }
    };
    auto save_output = [batch_size, stride, output_a] __device__ (scalar_t* s_input) mutable {
      const int b = blockIdx.x * blockDim.y + threadIdx.y;
      const int s = blockIdx.z;
      if (b < batch_size) {
        for (int i = threadIdx.x; i < stride * 2; i += blockDim.x) {
          output_a[b][s][i] = s_input[i + threadIdx.y * stride * 2];
        }
      }
    };
    bbt_ortho_multiply_untied_cuda_kernel<scalar_t>
      <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_cos_a, twiddle_sin_a, load_input,
                                                             save_output, log_stride, batch_size, nblocks);
  });
  AT_CHECK(cudaGetLastError() == cudaSuccess,
     "bbt_ortho_multiply_untied_cuda failed with error code ",
     cudaGetLastError());
}

template <typename scalar_t, typename accscalar_t,
          typename Function0, typename Function1, typename Function2>
__global__ void bbt_ortho_multiply_untied_backward_cuda_kernel(const CudaAcsr<scalar_t, 3> twiddle_cos_a,
                                                               const CudaAcsr<scalar_t, 3> twiddle_sin_a,
                                                               Function0 load_output,
                                                               Function1 load_grad,
                                                               CudaAcsr<scalar_t, 3> d_twiddle_a,
                                                               Function2 save_d_input,
                                                               int log_max_stride,
                                                               int batch_size,
                                                               int nblocks) {
  const int s = blockIdx.y + gridDim.y * blockIdx.z;  // For conv2d bbt_ortho as well
  const int max_stride = 1 << log_max_stride;
  const int input_base_idx = 0;
  __shared__ scalar_t s_output[ELEMENTARY_SIZE * 2];
  __shared__ scalar_t s_grad[ELEMENTARY_SIZE * 2];
  __shared__ scalar_t s_twiddle[ELEMENTARY_SIZE][2];
  __shared__ accscalar_t s_d_twiddle[ELEMENTARY_SIZE];
  int b = blockIdx.x * blockDim.y + threadIdx.y;
  int tid_x = threadIdx.x;
  int tid_y = threadIdx.y;
  load_output(s_output);
  load_grad(s_grad);
  for (int block = nblocks - 1; block >= 0; --block) {
    for (int idx = 2 * (log_max_stride + 1) - 1; idx >= 0; --idx) {
      int log_stride = idx <= log_max_stride ? log_max_stride - idx : idx - log_max_stride - 1;
      int stride = 1 << log_stride;
      // tid_y == 0 is writing (atomicAdd) so tid_y == -1 can do the reading, instead of having to wait for tid_y == 0
      if (tid_y == blockDim.y - 1) {
        s_twiddle[tid_x][0] = twiddle_cos_a[s][idx + block * 2 * (log_max_stride + 1)][input_base_idx / 2 + tid_x];
        s_twiddle[tid_x][1] = twiddle_sin_a[s][idx + block * 2 * (log_max_stride + 1)][input_base_idx / 2 + tid_x];
      }
      int low_order_bits = tid_x & (stride - 1);  // int low_order_bits = tid_x % stride;
      int pos_x = 2 * (tid_x - low_order_bits) + low_order_bits;
      int pos_y = tid_y * max_stride * 2;
      int pos = pos_x + pos_y;
      __syncthreads();
      const scalar_t twiddle_val[2] = {s_twiddle[tid_x][0], s_twiddle[tid_x][1]};
      if (b < batch_size) {
        const scalar_t grad_val[2] = {s_grad[pos], s_grad[pos + stride]};
        s_grad[pos] = twiddle_val[0] * grad_val[0] + twiddle_val[1] * grad_val[1];
        s_grad[pos + stride] = -twiddle_val[1] * grad_val[0] + twiddle_val[0] * grad_val[1];
        const scalar_t output_val[2] = {s_output[pos], s_output[pos + stride]};
        const scalar_t input_val[2] = {twiddle_val[0] * output_val[0] + twiddle_val[1] * output_val[1],
                                      -twiddle_val[1] * output_val[0] + twiddle_val[0] * output_val[1]};
        s_output[pos] = input_val[0];
        s_output[pos + stride] = input_val[1];
        s_d_twiddle[tid_x + tid_y * max_stride]
          = (grad_val[0] * input_val[0] + grad_val[1] * input_val[1]) * (-twiddle_val[1])
          + (-grad_val[0] * input_val[1] + grad_val[1] * input_val[0]) * twiddle_val[0];
      }
      __syncthreads();
      if (tid_y == 0) {
        accscalar_t d_twiddle_val = 0;
        for (int i = 0; i < blockDim.y; ++i) {
          if (blockIdx.x * blockDim.y + i < batch_size) {
            d_twiddle_val += s_d_twiddle[tid_x + i * max_stride];
          }
        }
        atomicAdd(&d_twiddle_a[s][idx + block * 2 * (log_max_stride + 1)][input_base_idx / 2 + tid_x], d_twiddle_val);
      }
    }
  }
  save_d_input(s_grad);
}

void bbt_ortho_multiply_untied_backward_cuda(const at::Tensor& twiddle_cos, const at::Tensor& twiddle_sin, const at::Tensor& output,
                                             const at::Tensor& grad, at::Tensor& d_twiddle, at::Tensor& d_input) {
  int batch_size = output.size(0);
  const int nstack = output.size(1);
  const int n = output.size(2);
  const int log_n = int(log2((double) n));
  int nblocks = twiddle_cos.size(1) / (2 * log_n);
  AT_DISPATCH_FLOATING_TYPES(output.scalar_type(), "bbt_ortho_multiply_untied_backward_cuda", [&] {
    using accscalar_t = at::acc_type<scalar_t, true>;
    const auto twiddle_cos_a = twiddle_cos.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>();
    const auto twiddle_sin_a = twiddle_sin.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>();
    const auto output_a = output.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>();
    const auto grad_a = grad.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>();
    auto d_twiddle_a = d_twiddle.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>();
    auto d_input_a = d_input.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>();
    int stride = std::min<int>(ELEMENTARY_SIZE, n / 2);
    int log_stride = int(log2((double) stride));
    dim3 block(stride, div_up(MAX_BLOCK_SIZE, stride * 2));
    dim3 grid(div_up(batch_size, block.y), 1, nstack);
    auto load_output = [batch_size, stride, output_a] __device__ (scalar_t* s_output) {
      const int b = blockIdx.x * blockDim.y + threadIdx.y;
      const int s = blockIdx.z;
      if (b < batch_size) {
        for (int i = threadIdx.x; i < stride * 2; i += blockDim.x) {
          s_output[i + threadIdx.y * stride * 2] = output_a[b][s][i];
        }
      }
    };
    auto load_grad = [batch_size, stride, grad_a] __device__ (scalar_t* s_grad) {
      const int b = blockIdx.x * blockDim.y + threadIdx.y;
      const int s = blockIdx.z;
      if (b < batch_size) {
        for (int i = threadIdx.x; i < stride * 2; i += blockDim.x) {
          s_grad[i + threadIdx.y * stride * 2] = grad_a[b][s][i];
        }
      }
    };
    auto save_d_input = [batch_size, stride, d_input_a] __device__ (scalar_t* s_grad) mutable {
      const int b = blockIdx.x * blockDim.y + threadIdx.y;
      const int s = blockIdx.z;
      if (b < batch_size) {
        for (int i = threadIdx.x; i < stride * 2; i += blockDim.x) {
          d_input_a[b][s][i] = s_grad[i + threadIdx.y * stride * 2];
        }
      }
    };
    bbt_ortho_multiply_untied_backward_cuda_kernel<scalar_t, accscalar_t>
      <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_cos_a, twiddle_sin_a, load_output, load_grad, d_twiddle_a, save_d_input, log_stride, batch_size, nblocks);
  });
  AT_CHECK(cudaGetLastError() == cudaSuccess,
     "bbt_ortho_multiply_untied_backward_cuda failed with error code ",
     cudaGetLastError());
}

template <typename scalar_t, bool increasing_stride, bool return_intermediates>
__global__ void butterfly_conv2d_cuda_kernel(const at::PackedTensorAccessor<scalar_t, 5> twiddle_a,
                                             const at::PackedTensorAccessor<scalar_t, 4> input_a,
                                             at::PackedTensorAccessor<scalar_t, 4> output_a,
                                             int log_max_stride,
                                             int log_n,
                                             int kernel_size,
                                             int padding,
                                             int h_out,
                                             int w_out) {
  const int batch_size = output_a.size(1);
  const int stack = blockIdx.z;
  const int s = blockIdx.y + gridDim.y * stack;
  const int max_stride = 1 << log_max_stride;
  // base index always 0
  const int input_base_idx = 0;
  const int h_in = input_a.size(2);
  const int w_in = input_a.size(3);
  __shared__ scalar_t s_input[ELEMENTARY_SIZE * 2];
  __shared__ scalar_t s_twiddle[ELEMENTARY_SIZE][2][2];
  int b = blockIdx.x * blockDim.y + threadIdx.y;
  const int patch_idx = b % (h_out * w_out);
  const int batch_idx = b / (h_out * w_out);
  int first_idx = increasing_stride ? 0 : log_n - 1 - log_max_stride;
  if (b < batch_size) {
    for (int t = threadIdx.x; t < max_stride * 2; t += blockDim.x) {
      // get index into patch
      int k_i = stack / kernel_size;
      int k_j = stack % kernel_size;
      // get patch index into full matrix
      int p_i = (patch_idx) / w_out;
      int p_j = (patch_idx) % (w_out);
      // combine indices and adjust for padding
      int i = k_i + p_i - padding;
      int j = k_j + p_j - padding;
      if (i >= w_in or j >= h_in or i < 0 or j < 0) s_input[t + threadIdx.y * max_stride * 2] = 0;
      else{
        s_input[t + threadIdx.y * max_stride * 2] = input_a[batch_idx][input_base_idx + t][i][j];
        // load input into first idx of output for backward pass
        // we allocated this memory already so shouldn't affect too much
        output_a[0][b][s][input_base_idx + t] = s_input[t + threadIdx.y * max_stride * 2];
      }
    }
  }
  int tid_x = threadIdx.x;
  int tid_y = threadIdx.y;
  for (int idx = first_idx; idx <= first_idx + log_max_stride; ++idx) {
    int log_stride = increasing_stride ? idx : log_n - 1 - idx;
    int stride = 1 << log_stride;
    if (tid_y == 0) {
      s_twiddle[tid_x][0][0] = twiddle_a[s][log_stride][input_base_idx / 2 + tid_x][0][0];
      s_twiddle[tid_x][0][1] = twiddle_a[s][log_stride][input_base_idx / 2 + tid_x][0][1];
      s_twiddle[tid_x][1][0] = twiddle_a[s][log_stride][input_base_idx / 2 + tid_x][1][0];
      s_twiddle[tid_x][1][1] = twiddle_a[s][log_stride][input_base_idx / 2 + tid_x][1][1];
    }
    int low_order_bits = tid_x & (stride - 1);  // int low_order_bits = tid_x % stride;
    int pos_x = 2 * (tid_x - low_order_bits) + low_order_bits;
    int pos_y = tid_y * max_stride * 2;
    int pos = pos_x + pos_y;
    __syncthreads();
    const scalar_t twiddle_val[2][2] = {{s_twiddle[tid_x][0][0], s_twiddle[tid_x][0][1]},
                                        {s_twiddle[tid_x][1][0], s_twiddle[tid_x][1][1]}};
    __syncthreads();  // otherwise some thread might go back to writing to s_twiddle before other thread can read
    if (b < batch_size) {
      const scalar_t input_val[2] = {s_input[pos], s_input[pos + stride]};
      s_input[pos] = twiddle_val[0][0] * input_val[0] + twiddle_val[0][1] * input_val[1];
      s_input[pos + stride] = twiddle_val[1][0] * input_val[0] + twiddle_val[1][1] * input_val[1];
      if (return_intermediates || idx == first_idx + log_max_stride) {
        output_a[idx+1][b][s][input_base_idx + pos_x] = s_input[pos];
        output_a[idx+1][b][s][input_base_idx + pos_x + stride] = s_input[pos + stride];
      }
    }
  }
}

void butterfly_conv2d_cuda(const at::Tensor& twiddle,
    const at::Tensor& input, at::Tensor& output,
    const int kernel_size, const int padding,
    const int h_out, const int w_out, bool increasing_stride,
    bool return_intermediates)
{
  const int b_in = input.size(0);
  const int n = input.size(1); /*c*/
  const int nstack = twiddle.size(0);
  const int stack = kernel_size*kernel_size;
  const int log_n = int(log2((double) n));
  const int batch_size = output.size(1);
  AT_DISPATCH_FLOATING_TYPES(output.scalar_type(),
    "butterfly_conv2d_cuda", [&] {
      const auto twiddle_a = twiddle.packed_accessor<scalar_t, 5>();
      // batch_size, c, h, w
      const auto input_a = input.packed_accessor<scalar_t, 4>();
      // log c_in, h*w*batch_size, nstack, c_in
      auto output_a = output.packed_accessor<scalar_t, 4>();
      // assume in_channels <= 1024
      int stride = std::min<int>(ELEMENTARY_SIZE, n / 2);
      int log_stride = int(log2((double) stride));
      // to support out_channels > in_channels
      int c_out_ratio = nstack / stack;
      // dim3 block(stride);
      // dim3 grid(batch_size, c_out_ratio, stack);
      dim3 block(stride, div_up(MAX_BLOCK_SIZE, stride * 2));
      dim3 grid(div_up(batch_size, block.y), c_out_ratio, stack);
      if (increasing_stride) {
        return_intermediates ? butterfly_conv2d_cuda_kernel<scalar_t, true, true>
        <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, input_a,
          output_a, log_stride, log_n, kernel_size, padding, h_out, w_out)
                           : butterfly_conv2d_cuda_kernel<scalar_t, true, false>
        <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, input_a,
          output_a, log_stride, log_n, kernel_size, padding, h_out, w_out);
      }
      else {
        return_intermediates ? butterfly_conv2d_cuda_kernel<scalar_t, false, true>
        <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, input_a,
          output_a, log_stride, log_n, kernel_size, padding, h_out, w_out)
                           : butterfly_conv2d_cuda_kernel<scalar_t, false, false>
        <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, input_a,
          output_a, log_stride, log_n, kernel_size, padding, h_out, w_out);
      }
  });
  AT_CHECK(cudaGetLastError() == cudaSuccess,
     "butterfly_conv2d_cuda failed with error code ",
     cudaGetLastError());
}

template <typename scalar_t, typename accscalar_t, bool increasing_stride>
__global__ void butterfly_conv2d_backward_cuda_kernel(
    const at::PackedTensorAccessor<scalar_t, 3> grad_a,
    const at::PackedTensorAccessor<scalar_t, 5> twiddle_a,
    const at::PackedTensorAccessor<scalar_t, 4> output_a,
    at::PackedTensorAccessor<scalar_t, 5> d_twiddle_a,
    at::PackedTensorAccessor<scalar_t, 4> d_input_a,
    int log_max_stride,
    int log_n,
    int kernel_size,
    int padding,
    int h_out,
    int w_out) {
  const int batch_size = output_a.size(1);
  const int stack = blockIdx.z;
  const int s = blockIdx.y + gridDim.y * stack;
  // base index always 0
  const int input_base_idx = 0;
  const int h_in = d_input_a.size(2);
  const int w_in = d_input_a.size(3);
  const int max_stride = 1 << log_max_stride;
  __shared__ scalar_t s_grad[ELEMENTARY_SIZE * 2];
  __shared__ accscalar_t s_twiddle[ELEMENTARY_SIZE][2][2];  // Use accscalar_t instead of scalar_t since we'll reuse the storage for s_d_twiddle
  accscalar_t* s_d_twiddle = (accscalar_t *)&s_twiddle[0][0][0];  // Reusing the same storage as s_twiddle, have to be careful if we change the implemetnation.
  int b = blockIdx.x * blockDim.y + threadIdx.y;
  if (b < batch_size) {
    for (int i = threadIdx.x; i < max_stride * 2; i += blockDim.x) {
      s_grad[i + threadIdx.y * max_stride * 2] = grad_a[b][s][input_base_idx + i];
    }
  }
  int tid_x = threadIdx.x;
  int tid_y = threadIdx.y;
  int first_idx = increasing_stride ? 0 : log_n - 1 - log_max_stride;
  for (int idx = first_idx + log_max_stride; idx >= first_idx; --idx) {
    int log_stride = increasing_stride ? idx : log_n - 1 - idx;
    int stride = 1 << log_stride;
    if (tid_y == 0) {
      s_twiddle[tid_x][0][0] = twiddle_a[s][log_stride][input_base_idx / 2 + tid_x][0][0];
      s_twiddle[tid_x][0][1] = twiddle_a[s][log_stride][input_base_idx / 2 + tid_x][0][1];
      s_twiddle[tid_x][1][0] = twiddle_a[s][log_stride][input_base_idx / 2 + tid_x][1][0];
      s_twiddle[tid_x][1][1] = twiddle_a[s][log_stride][input_base_idx / 2 + tid_x][1][1];
    }
    int low_order_bits = tid_x & (stride - 1);  // int low_order_bits = tid_x % stride;
    int pos_x = 2 * (tid_x - low_order_bits) + low_order_bits;
    int pos_y = tid_y * max_stride * 2;
    int pos = pos_x + pos_y;
    __syncthreads();
    const scalar_t twiddle_val[2][2] = {{s_twiddle[tid_x][0][0], s_twiddle[tid_x][0][1]},
                                        {s_twiddle[tid_x][1][0], s_twiddle[tid_x][1][1]}};
    // Don't need to sync here since we sync later at sum_strided_atomic, so no writing to s_twiddle can occur until then
    accscalar_t d_twiddle_val[2][2] = {{0, 0}, {0, 0}};
    if (b < batch_size) {
      const scalar_t grad_val[2] = {s_grad[pos], s_grad[pos + stride]};
      s_grad[pos] = twiddle_val[0][0] * grad_val[0] + twiddle_val[1][0] * grad_val[1];
      s_grad[pos + stride] = twiddle_val[0][1] * grad_val[0] + twiddle_val[1][1] * grad_val[1];
      const scalar_t input_val[2] = {output_a[idx][b][s][input_base_idx + pos_x],
                                     output_a[idx][b][s][input_base_idx + pos_x + stride]};
      d_twiddle_val[0][0] = grad_val[0] * input_val[0];
      d_twiddle_val[0][1] = grad_val[0] * input_val[1];
      d_twiddle_val[1][0] = grad_val[1] * input_val[0];
      d_twiddle_val[1][1] = grad_val[1] * input_val[1];
    }
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int nthreads = blockDim.x * blockDim.y;
    sum_strided_atomic(reinterpret_cast<accscalar_t (&)[4]>(d_twiddle_val), s_d_twiddle, max_stride, nthreads, tid);
    if (tid_y == 0) {
      atomicAdd(&d_twiddle_a[s][log_stride][input_base_idx / 2 + tid_x][0][0], s_d_twiddle[tid_x]);
      atomicAdd(&d_twiddle_a[s][log_stride][input_base_idx / 2 + tid_x][0][1], s_d_twiddle[tid_x + max_stride]);
      atomicAdd(&d_twiddle_a[s][log_stride][input_base_idx / 2 + tid_x][1][0], s_d_twiddle[tid_x + 2 * max_stride]);
      atomicAdd(&d_twiddle_a[s][log_stride][input_base_idx / 2 + tid_x][1][1], s_d_twiddle[tid_x + 3 * max_stride]);
    }
    __syncthreads();  // Otherwise s_d_twiddle will be overwritten with s_twiddle before some thread can read
  }
  if (b < batch_size) {
    const int patch_idx = b % (h_out * w_out);
    const int batch_idx = b / (h_out * w_out);
    for (int t = threadIdx.x; t < max_stride * 2; t += blockDim.x) {
      // map back to b, c, h, w
      // get index into patch
      int k_i = stack / kernel_size; // stack / kernel_size
      int k_j = stack % kernel_size; // stack % kernel_size
      // get patch index into full matrix
      int p_i = (patch_idx) / w_out;
      int p_j = (patch_idx) % (w_out);
      // combine indices and adjust for padding
      int i = k_i + p_i - padding;
      int j = k_j + p_j - padding;
      // this needs to be atomic because input is reused in forward pass
      // with out_channels > in_channels and for each entry of the patch
      if (i < w_in && j < h_in && i >= 0 && j >= 0) {
        atomicAdd(&d_input_a[batch_idx][input_base_idx + t][i][j], s_grad[t + threadIdx.y * max_stride * 2]);
      }
    }
  }
}

void butterfly_conv2d_backward_cuda(const at::Tensor&grad, const at::Tensor& twiddle,
  const at::Tensor& output, at::Tensor& d_twiddle, at::Tensor& d_input,
  const int kernel_size, const int padding,
  const int h_out, const int w_out,
  bool increasing_stride) {
  const int batch_size = output.size(1);
  const int nstack = twiddle.size(0);
  const int stack = kernel_size*kernel_size;
  const int n = d_input.size(1); // c_in
  const int log_n = int(log2((double) n));
  AT_DISPATCH_FLOATING_TYPES(output.scalar_type(),
  "butterfly_conv2d_backward_cuda", [&] {
    using accscalar_t = at::acc_type<scalar_t, true>;
    const auto grad_a = grad.packed_accessor<scalar_t, 3>();
    const auto twiddle_a = twiddle.packed_accessor<scalar_t, 5>();
    const auto output_a = output.packed_accessor<scalar_t, 4>();
    auto d_twiddle_a = d_twiddle.packed_accessor<scalar_t, 5>();
    auto d_input_a = d_input.packed_accessor<scalar_t, 4>();
    int stride = std::min<int>(ELEMENTARY_SIZE, n / 2);
    int log_stride = int(log2((double) stride));
    // to support out_channels > in_channels
    int c_out_ratio = nstack / stack;
    // dim3 block(stride);
    // dim3 grid(batch_size, c_out_ratio, stack);
    dim3 block(stride, div_up(MAX_BLOCK_SIZE, stride * 2));
    dim3 grid(div_up(batch_size, block.y), c_out_ratio, stack);
    increasing_stride ?
      butterfly_conv2d_backward_cuda_kernel<scalar_t, accscalar_t, true>
      <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        grad_a, twiddle_a, output_a, d_twiddle_a, d_input_a, log_stride,
        log_n, kernel_size, padding, h_out, w_out) :
      butterfly_conv2d_backward_cuda_kernel<scalar_t, accscalar_t, false>
        <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
          grad_a, twiddle_a, output_a, d_twiddle_a, d_input_a, log_stride,
          log_n, kernel_size, padding, h_out, w_out);
  });
  AT_CHECK(cudaGetLastError() == cudaSuccess,
     "butterfly_conv2d_backward_cuda failed with error code ",
     cudaGetLastError());
}

void butterfly_conv2d_forward_backward_cuda(const at::Tensor& twiddle,
    const at::Tensor& input, const at::Tensor&grad,
    at::Tensor& d_twiddle, at::Tensor& d_input,
    const int kernel_size, const int padding, const int h_out, const int w_out,
    bool increasing_stride) {
  const int batch_size = grad.size(0); // b_out = b_in * h_out * w_out
  const int nstack = twiddle.size(0);
  const int stack = kernel_size * kernel_size;
  const int n = d_input.size(1); // c_in
  const int log_n = int(log2((double) n));
  const int c_out_ratio = nstack / stack;
  const int h_in = input.size(2);
  const int w_in = input.size(3);
  AT_DISPATCH_FLOATING_TYPES(grad.scalar_type(),
  "butterfly_conv2d_forward_backward_cuda", [&] {
    using accscalar_t = at::acc_type<scalar_t, true>;
    const auto twiddle_a = twiddle.packed_accessor<scalar_t, 5, at::RestrictPtrTraits, int32_t>();
    const auto input_a = input.packed_accessor<scalar_t, 4, at::RestrictPtrTraits, int32_t>();
    const auto grad_a = grad.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>();
    auto d_twiddle_a = d_twiddle.packed_accessor<scalar_t, 5, at::RestrictPtrTraits, int32_t>();
    auto d_input_a = d_input.packed_accessor<scalar_t, 4, at::RestrictPtrTraits, int32_t>();
    int stride = std::min<int>(ELEMENTARY_SIZE, n / 2);
    int log_stride = int(log2((double) stride));
    dim3 block(stride, div_up(MAX_BLOCK_SIZE, stride * 2));
    dim3 grid(div_up(batch_size, block.y), c_out_ratio, stack);
    auto load_input = [batch_size, stride, input_a, kernel_size, padding, h_out, w_out, h_in, w_in] __device__ (scalar_t* s_input) {
      const int b = blockIdx.x * blockDim.y + threadIdx.y;
      const int stack = blockIdx.z;
      const int patch_idx = b % (h_out * w_out);
      const int batch_idx = b / (h_out * w_out);
      if (b < batch_size) {
        for (int t = threadIdx.x; t < stride * 2; t += blockDim.x) {
          // get index into patch
          int k_i = stack / kernel_size;
          int k_j = stack % kernel_size;
          // get patch index into full matrix
          int p_i = (patch_idx) / w_out;
          int p_j = (patch_idx) % (w_out);
          // combine indices and adjust for padding
          int i = k_i + p_i - padding;
          int j = k_j + p_j - padding;
          if (i >= w_in or j >= h_in or i < 0 or j < 0) s_input[t + threadIdx.y * stride * 2] = 0;
          else{
            s_input[t + threadIdx.y * stride * 2] = input_a[batch_idx][t][i][j];
          }
        }
      }
    };
    auto load_grad = [batch_size, stride, grad_a] __device__ (scalar_t* s_grad) {
      const int b = blockIdx.x * blockDim.y + threadIdx.y;
      const int stack = blockIdx.z;
      const int s = blockIdx.y + gridDim.y * stack;
      if (b < batch_size) {
        for (int i = threadIdx.x; i < stride * 2; i += blockDim.x) {
          s_grad[i + threadIdx.y * stride * 2] = grad_a[b][s][i];
        }
      }
    };
    auto save_d_input = [batch_size, stride, d_input_a, kernel_size, padding, h_out, w_out, h_in, w_in] __device__ (scalar_t* s_grad) mutable {
      const int b = blockIdx.x * blockDim.y + threadIdx.y;
      const int stack = blockIdx.z;
      const int patch_idx = b % (h_out * w_out);
      const int batch_idx = b / (h_out * w_out);
      if (b < batch_size) {
        for (int t = threadIdx.x; t < stride * 2; t += blockDim.x) {
          // map back to b, c, h, w
          // get index into patch
          int k_i = stack / kernel_size;
          int k_j = stack % kernel_size;
          // get patch index into full matrix
          int p_i = (patch_idx) / w_out;
          int p_j = (patch_idx) % (w_out);
          // combine indices and adjust for padding
          int i = k_i + p_i - padding;
          int j = k_j + p_j - padding;
          if (i < w_in && j < h_in && i >= 0 && j >= 0) {
            atomicAdd(&d_input_a[batch_idx][t][i][j], s_grad[t + threadIdx.y * stride * 2]);
          }
        }
      }
    };
    switch (log_stride)
      {
      case 0:
        increasing_stride ? butterfly_multiply_untied_forward_backward_cuda_kernel<scalar_t, accscalar_t, true, 0>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, load_input, load_grad, d_twiddle_a, save_d_input, batch_size)
                          : butterfly_multiply_untied_forward_backward_cuda_kernel<scalar_t, accscalar_t, false, 0>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, load_input, load_grad, d_twiddle_a, save_d_input, batch_size); break;
      case 1:
        increasing_stride ? butterfly_multiply_untied_forward_backward_cuda_kernel<scalar_t, accscalar_t, true, 1>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, load_input, load_grad, d_twiddle_a, save_d_input, batch_size)
                          : butterfly_multiply_untied_forward_backward_cuda_kernel<scalar_t, accscalar_t, false, 1>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, load_input, load_grad, d_twiddle_a, save_d_input, batch_size); break;
      case 2:
        increasing_stride ? butterfly_multiply_untied_forward_backward_cuda_kernel<scalar_t, accscalar_t, true, 2>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, load_input, load_grad, d_twiddle_a, save_d_input, batch_size)
                          : butterfly_multiply_untied_forward_backward_cuda_kernel<scalar_t, accscalar_t, false, 2>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, load_input, load_grad, d_twiddle_a, save_d_input, batch_size); break;
      case 3:
        increasing_stride ? butterfly_multiply_untied_forward_backward_cuda_kernel<scalar_t, accscalar_t, true, 3>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, load_input, load_grad, d_twiddle_a, save_d_input, batch_size)
                          : butterfly_multiply_untied_forward_backward_cuda_kernel<scalar_t, accscalar_t, false, 3>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, load_input, load_grad, d_twiddle_a, save_d_input, batch_size); break;
      case 4:
        increasing_stride ? butterfly_multiply_untied_forward_backward_cuda_kernel<scalar_t, accscalar_t, true, 4>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, load_input, load_grad, d_twiddle_a, save_d_input, batch_size)
                          : butterfly_multiply_untied_forward_backward_cuda_kernel<scalar_t, accscalar_t, false, 4>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, load_input, load_grad, d_twiddle_a, save_d_input, batch_size); break;
      case 5:
        increasing_stride ? butterfly_multiply_untied_forward_backward_cuda_kernel<scalar_t, accscalar_t, true, 5>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, load_input, load_grad, d_twiddle_a, save_d_input, batch_size)
                          : butterfly_multiply_untied_forward_backward_cuda_kernel<scalar_t, accscalar_t, false, 5>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, load_input, load_grad, d_twiddle_a, save_d_input, batch_size); break;
      case 6:
        increasing_stride ? butterfly_multiply_untied_forward_backward_cuda_kernel<scalar_t, accscalar_t, true, 6>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, load_input, load_grad, d_twiddle_a, save_d_input, batch_size)
                          : butterfly_multiply_untied_forward_backward_cuda_kernel<scalar_t, accscalar_t, false, 6>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, load_input, load_grad, d_twiddle_a, save_d_input, batch_size); break;
      case 7:
        increasing_stride ? butterfly_multiply_untied_forward_backward_cuda_kernel<scalar_t, accscalar_t, true, 7>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, load_input, load_grad, d_twiddle_a, save_d_input, batch_size)
                          : butterfly_multiply_untied_forward_backward_cuda_kernel<scalar_t, accscalar_t, false, 7>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, load_input, load_grad, d_twiddle_a, save_d_input, batch_size); break;
      case 8:
        increasing_stride ? butterfly_multiply_untied_forward_backward_cuda_kernel<scalar_t, accscalar_t, true, 8>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, load_input, load_grad, d_twiddle_a, save_d_input, batch_size)
                          : butterfly_multiply_untied_forward_backward_cuda_kernel<scalar_t, accscalar_t, false, 8>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, load_input, load_grad, d_twiddle_a, save_d_input, batch_size); break;
      case 9:
        increasing_stride ? butterfly_multiply_untied_forward_backward_cuda_kernel<scalar_t, accscalar_t, true, 9>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, load_input, load_grad, d_twiddle_a, save_d_input, batch_size)
                          : butterfly_multiply_untied_forward_backward_cuda_kernel<scalar_t, accscalar_t, false, 9>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, load_input, load_grad, d_twiddle_a, save_d_input, batch_size); break;
      }
  });
  AT_CHECK(cudaGetLastError() == cudaSuccess,
     "butterfly_conv2d_forward_backward_cuda failed with error code ",
     cudaGetLastError());
}

void bbt_conv2d_cuda(const at::Tensor& twiddle,
    const at::Tensor& input, at::Tensor& output,
    const int kernel_size, const int padding,
    const int h_out, const int w_out)
{
  const int b_in = input.size(0);
  const int n = input.size(1); /*c*/
  const int nstack = twiddle.size(0);
  const int stack = kernel_size*kernel_size;
  const int log_n = int(log2((double) n));
  int nblocks = twiddle.size(1) / (2 * log_n);
  int batch_size = output.size(0);
  const int h_in = input.size(2);
  const int w_in = input.size(3);
  AT_DISPATCH_FLOATING_TYPES(output.scalar_type(),
    "bbt_conv2d_cuda", [&] {
      const auto twiddle_a = twiddle.packed_accessor<scalar_t, 5, at::RestrictPtrTraits, int32_t>();
      // batch_size, c, h, w
      const auto input_a = input.packed_accessor<scalar_t, 4, at::RestrictPtrTraits, int32_t>();
      // h*w*batch_size, nstack, c_in
      auto output_a = output.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>();
      // assume in_channels <= 1024
      int stride = std::min<int>(ELEMENTARY_SIZE, n / 2);
      int log_stride = int(log2((double) stride));
      // to support out_channels > in_channels
      int c_out_ratio = nstack / stack;
      // dim3 block(stride);
      // dim3 grid(batch_size, c_out_ratio, stack);
      dim3 block(stride, div_up(MAX_BLOCK_SIZE, stride * 2));
      dim3 grid(div_up(batch_size, block.y), c_out_ratio, stack);
      auto load_input = [batch_size, stride, input_a, kernel_size, padding, h_out, w_out, h_in, w_in] __device__ (scalar_t* s_input) {
        const int b = blockIdx.x * blockDim.y + threadIdx.y;
        const int stack = blockIdx.z;
        const int patch_idx = b % (h_out * w_out);
        const int batch_idx = b / (h_out * w_out);
        if (b < batch_size) {
          for (int t = threadIdx.x; t < stride * 2; t += blockDim.x) {
            // get index into patch
            int k_i = stack / kernel_size;
            int k_j = stack % kernel_size;
            // get patch index into full matrix
            int p_i = (patch_idx) / w_out;
            int p_j = (patch_idx) % (w_out);
            // combine indices and adjust for padding
            int i = k_i + p_i - padding;
            int j = k_j + p_j - padding;
            if (i >= w_in or j >= h_in or i < 0 or j < 0) s_input[t + threadIdx.y * stride * 2] = 0;
            else{
              s_input[t + threadIdx.y * stride * 2] = input_a[batch_idx][t][i][j];
            }
          }
        }
      };
      auto save_output = [batch_size, stride, output_a] __device__ (scalar_t* s_input) mutable {
        const int b = blockIdx.x * blockDim.y + threadIdx.y;
        const int stack = blockIdx.z;
        const int s = blockIdx.y + gridDim.y * stack;
        if (b < batch_size) {
          for (int i = threadIdx.x; i < stride * 2; i += blockDim.x) {
            output_a[b][s][i] = s_input[i + threadIdx.y * stride * 2];
          }
        }
      };
      bbt_multiply_untied_cuda_kernel<scalar_t>
        <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, load_input, save_output, log_stride, batch_size, nblocks) ;
  });
  AT_CHECK(cudaGetLastError() == cudaSuccess,
     "bbt_conv2d_cuda failed with error code ",
     cudaGetLastError());
}

void bbt_conv2d_forward_backward_cuda(const at::Tensor& twiddle,
    const at::Tensor& input, const at::Tensor&grad,
    at::Tensor& d_twiddle, at::Tensor& d_input,
    const int kernel_size, const int padding, const int h_out, const int w_out) {
  int batch_size = grad.size(0); // b_out = b_in * h_out * w_out
  const int nstack = twiddle.size(0);
  const int stack = kernel_size * kernel_size;
  const int n = d_input.size(1); // c_in
  const int log_n = int(log2((double) n));
  int nblocks = twiddle.size(1) / (2 * log_n);
  const int c_out_ratio = nstack / stack;
  const int h_in = input.size(2);
  const int w_in = input.size(3);
  AT_DISPATCH_FLOATING_TYPES(grad.scalar_type(),
  "bbt_conv2d_forward_backward_cuda", [&] {
    using accscalar_t = at::acc_type<scalar_t, true>;
    const auto twiddle_a = twiddle.packed_accessor<scalar_t, 5, at::RestrictPtrTraits, int32_t>();
    const auto input_a = input.packed_accessor<scalar_t, 4, at::RestrictPtrTraits, int32_t>();
    const auto grad_a = grad.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>();
    auto d_twiddle_a = d_twiddle.packed_accessor<scalar_t, 5, at::RestrictPtrTraits, int32_t>();
    auto d_input_a = d_input.packed_accessor<scalar_t, 4, at::RestrictPtrTraits, int32_t>();
    int stride = std::min<int>(ELEMENTARY_SIZE, n / 2);
    int log_stride = int(log2((double) stride));
    dim3 block(stride, div_up(MAX_BLOCK_SIZE, stride * 2));
    dim3 grid(div_up(batch_size, block.y), c_out_ratio, stack);
    auto load_input = [batch_size, stride, input_a, kernel_size, padding, h_out, w_out, h_in, w_in] __device__ (scalar_t* s_input) {
      const int b = blockIdx.x * blockDim.y + threadIdx.y;
      const int stack = blockIdx.z;
      const int patch_idx = b % (h_out * w_out);
      const int batch_idx = b / (h_out * w_out);
      if (b < batch_size) {
        for (int t = threadIdx.x; t < stride * 2; t += blockDim.x) {
          // get index into patch
          int k_i = stack / kernel_size;
          int k_j = stack % kernel_size;
          // get patch index into full matrix
          int p_i = (patch_idx) / w_out;
          int p_j = (patch_idx) % (w_out);
          // combine indices and adjust for padding
          int i = k_i + p_i - padding;
          int j = k_j + p_j - padding;
          if (i >= w_in or j >= h_in or i < 0 or j < 0) s_input[t + threadIdx.y * stride * 2] = 0;
          else{
            s_input[t + threadIdx.y * stride * 2] = input_a[batch_idx][t][i][j];
          }
        }
      }
    };
    auto load_grad = [batch_size, stride, grad_a] __device__ (scalar_t* s_grad) {
      const int b = blockIdx.x * blockDim.y + threadIdx.y;
      const int stack = blockIdx.z;
      const int s = blockIdx.y + gridDim.y * stack;
      if (b < batch_size) {
        for (int i = threadIdx.x; i < stride * 2; i += blockDim.x) {
          s_grad[i + threadIdx.y * stride * 2] = grad_a[b][s][i];
        }
      }
    };
    auto save_d_input = [batch_size, stride, d_input_a, kernel_size, padding, h_out, w_out, h_in, w_in] __device__ (scalar_t* s_grad) mutable {
      const int b = blockIdx.x * blockDim.y + threadIdx.y;
      const int stack = blockIdx.z;
      const int patch_idx = b % (h_out * w_out);
      const int batch_idx = b / (h_out * w_out);
      if (b < batch_size) {
        for (int t = threadIdx.x; t < stride * 2; t += blockDim.x) {
          // map back to b, c, h, w
          // get index into patch
          int k_i = stack / kernel_size;
          int k_j = stack % kernel_size;
          // get patch index into full matrix
          int p_i = (patch_idx) / w_out;
          int p_j = (patch_idx) % (w_out);
          // combine indices and adjust for padding
          int i = k_i + p_i - padding;
          int j = k_j + p_j - padding;
          if (i < w_in && j < h_in && i >= 0 && j >= 0) {
            atomicAdd(&d_input_a[batch_idx][t][i][j], s_grad[t + threadIdx.y * stride * 2]);
          }
        }
      }
    };
    switch (nblocks)
      {
      case 1:
        bbt_multiply_untied_forward_backward_cuda_kernel<scalar_t, accscalar_t, 1>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, load_input, load_grad, d_twiddle_a, save_d_input, log_stride, batch_size); break;
      case 2:
        bbt_multiply_untied_forward_backward_cuda_kernel<scalar_t, accscalar_t, 2>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, load_input, load_grad, d_twiddle_a, save_d_input, log_stride, batch_size); break;
      case 3:
        bbt_multiply_untied_forward_backward_cuda_kernel<scalar_t, accscalar_t, 3>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, load_input, load_grad, d_twiddle_a, save_d_input, log_stride, batch_size); break;
      case 4:
        bbt_multiply_untied_forward_backward_cuda_kernel<scalar_t, accscalar_t, 4>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, load_input, load_grad, d_twiddle_a, save_d_input, log_stride, batch_size); break;
      case 5:
        bbt_multiply_untied_forward_backward_cuda_kernel<scalar_t, accscalar_t, 5>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, load_input, load_grad, d_twiddle_a, save_d_input, log_stride, batch_size); break;
      case 6:
        bbt_multiply_untied_forward_backward_cuda_kernel<scalar_t, accscalar_t, 6>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, load_input, load_grad, d_twiddle_a, save_d_input, log_stride, batch_size); break;
      case 7:
        bbt_multiply_untied_forward_backward_cuda_kernel<scalar_t, accscalar_t, 7>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, load_input, load_grad, d_twiddle_a, save_d_input, log_stride, batch_size); break;
      case 8:
        bbt_multiply_untied_forward_backward_cuda_kernel<scalar_t, accscalar_t, 8>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, load_input, load_grad, d_twiddle_a, save_d_input, log_stride, batch_size); break;
      case 9:
        bbt_multiply_untied_forward_backward_cuda_kernel<scalar_t, accscalar_t, 9>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, load_input, load_grad, d_twiddle_a, save_d_input, log_stride, batch_size); break;
      case 10:
        bbt_multiply_untied_forward_backward_cuda_kernel<scalar_t, accscalar_t, 10>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, load_input, load_grad, d_twiddle_a, save_d_input, log_stride, batch_size); break;
      case 11:
        bbt_multiply_untied_forward_backward_cuda_kernel<scalar_t, accscalar_t, 11>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, load_input, load_grad, d_twiddle_a, save_d_input, log_stride, batch_size); break;
      case 12:
        bbt_multiply_untied_forward_backward_cuda_kernel<scalar_t, accscalar_t, 12>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, load_input, load_grad, d_twiddle_a, save_d_input, log_stride, batch_size); break;
      case 13:
        bbt_multiply_untied_forward_backward_cuda_kernel<scalar_t, accscalar_t, 13>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, load_input, load_grad, d_twiddle_a, save_d_input, log_stride, batch_size); break;
      case 14:
        bbt_multiply_untied_forward_backward_cuda_kernel<scalar_t, accscalar_t, 14>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, load_input, load_grad, d_twiddle_a, save_d_input, log_stride, batch_size); break;
      }
  });
  AT_CHECK(cudaGetLastError() == cudaSuccess,
     "bbt_conv2d_forward_backward_cuda failed with error code ",
     cudaGetLastError());
}

template <typename scalar_t, bool increasing_stride, bool return_intermediates>
__global__ void butterfly_multiply_untied_svd_cuda_kernel(const at::PackedTensorAccessor<scalar_t, 5> twiddle_a,
                                                          at::PackedTensorAccessor<scalar_t, 4> output_a,
                                                          int log_max_stride,
                                                          int log_n) {
  const int batch_size = output_a.size(1);
  const int s = blockIdx.z;
  const int max_stride = 1 << log_max_stride;
  const int input_base_idx = blockIdx.y * blockDim.x * 2;
  __shared__ scalar_t s_input[ELEMENTARY_SIZE * 2];
  int b = blockIdx.x * blockDim.y + threadIdx.y;
  if (b < batch_size) {  // Currently we assume 1 batch per thread block, so all threads in the block should enter (otherwise deadlock)
    int first_idx = increasing_stride ? 0 : log_n - 1 - log_max_stride;
    for (int i = threadIdx.x; i < max_stride * 2; i += blockDim.x) {
      s_input[i] = output_a[first_idx][b][s][input_base_idx + i];
    }
    int i = threadIdx.x;
    for (int idx = first_idx; idx <= first_idx + log_max_stride; ++idx) {
      int log_stride = increasing_stride ? idx : log_n - 1 - idx;
      int stride = 1 << log_stride;
      int low_order_bits = i & (stride - 1);  // int low_order_bits = i % stride;
      int pos = 2 * (i - low_order_bits) + low_order_bits;
      const scalar_t twiddle_val[2][2] = {{twiddle_a[s][log_stride][input_base_idx / 2 + i][0][0], twiddle_a[s][log_stride][input_base_idx / 2 + i][0][1]},
                                          {twiddle_a[s][log_stride][input_base_idx / 2 + i][1][0], twiddle_a[s][log_stride][input_base_idx / 2 + i][1][1]}};
      const scalar_t sin_theta = thc_sin(twiddle_val[0][0]), cos_theta = thc_cos(twiddle_val[0][0]);
      const scalar_t sin_phi = thc_sin(twiddle_val[0][1]), cos_phi = thc_cos(twiddle_val[0][1]);
      __syncthreads();
      const scalar_t input_val[2] = {s_input[pos], s_input[pos + stride]};
      scalar_t temp[2];
      thrust::tie(temp[0], temp[1]) = mult2x2(cos_phi, -sin_phi, sin_phi, cos_phi, input_val[0], input_val[1]);
      thrust::tie(temp[0], temp[1]) = mult2x2(cos_theta, -sin_theta, sin_theta, cos_theta,
                                              temp[0] * twiddle_val[1][0], temp[1] * twiddle_val[1][1]);
      s_input[pos] = temp[0];
      s_input[pos + stride] = temp[1];
      if (return_intermediates || idx == first_idx + log_max_stride) {
        output_a[idx+1][b][s][input_base_idx + pos] = s_input[pos];
        output_a[idx+1][b][s][input_base_idx + pos + stride] = s_input[pos + stride];
      }
    }
  }
}

template <typename scalar_t, bool increasing_stride>
__global__ void butterfly_multiply_untied_svd_onestep_cuda_kernel(const at::PackedTensorAccessor<scalar_t, 5> twiddle_a,
                                                                  at::PackedTensorAccessor<scalar_t, 4> output_a,
                                                                  int log_stride,
                                                                  int log_n) {
  const int batch_size = output_a.size(1);
  const int s = blockIdx.z;
  const int idx = increasing_stride ? log_stride : (log_n - 1 - log_stride);  // Index to access output_a
  const int stride = 1 << log_stride;
  int i = blockIdx.y * blockDim.x + threadIdx.x;
  int low_order_bits = i & (stride - 1);  // int low_order_bits = i % stride;
  int pos = 2 * (i - low_order_bits) + low_order_bits;
  const scalar_t twiddle_val[2][2] = {{twiddle_a[s][log_stride][i][0][0], twiddle_a[s][log_stride][i][0][1]},
                                      {twiddle_a[s][log_stride][i][1][0], twiddle_a[s][log_stride][i][1][1]}};
  const scalar_t sin_theta = thc_sin(twiddle_val[0][0]), cos_theta = thc_cos(twiddle_val[0][0]);
  const scalar_t sin_phi = thc_sin(twiddle_val[0][1]), cos_phi = thc_cos(twiddle_val[0][1]);
  for (int b = blockIdx.x * blockDim.y + threadIdx.y; b < batch_size; b += blockDim.y * gridDim.x) {
    const scalar_t input_val[2] = {output_a[idx][b][s][pos], output_a[idx][b][s][pos + stride]};
    scalar_t temp[2];
    thrust::tie(temp[0], temp[1]) = mult2x2(cos_phi, -sin_phi, sin_phi, cos_phi, input_val[0], input_val[1]);
    thrust::tie(temp[0], temp[1]) = mult2x2(cos_theta, -sin_theta, sin_theta, cos_theta,
                                            temp[0] * twiddle_val[1][0], temp[1] * twiddle_val[1][1]);
    output_a[idx+1][b][s][pos] = temp[0];
    output_a[idx+1][b][s][pos + stride] = temp[1];
  }
}

void butterfly_multiply_untied_svd_cuda(const at::Tensor& twiddle, at::Tensor& output,
                                        bool increasing_stride, bool return_intermediates) {
  const int batch_size = output.size(1);
  const int nstack = twiddle.size(0);
  const int n = output.size(3);
  const int log_n = int(log2((double) n));
  AT_DISPATCH_FLOATING_TYPES(output.scalar_type(), "butterfly_multiply_untied_svd_cuda", [&] {
    const auto twiddle_a = twiddle.packed_accessor<scalar_t, 5>();
    auto output_a = output.packed_accessor<scalar_t, 4>();
    if (increasing_stride) {
      int stride = std::min<int>(ELEMENTARY_SIZE, n / 2);
      int log_stride = int(log2((double) stride));
      dim3 block(stride);
      dim3 grid(batch_size, div_up(n / 2, stride), nstack);
      return_intermediates ? butterfly_multiply_untied_svd_cuda_kernel<scalar_t, true, true>
        <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, output_a, log_stride, log_n)
                            : butterfly_multiply_untied_svd_cuda_kernel<scalar_t, true, false>
        <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, output_a, log_stride, log_n);
      for (log_stride++; log_stride <= log_n - 1; ++log_stride) {
        dim3 block(MAX_BLOCK_SIZE / 2);
        dim3 grid(div_up(batch_size, WORK_PER_THREAD), div_up(n / 2, MAX_BLOCK_SIZE / 2), nstack);
        butterfly_multiply_untied_svd_onestep_cuda_kernel<scalar_t, true>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, output_a, log_stride, log_n);
      }
    } else {
      int log_stride = log_n - 1;
      for (; (1 << log_stride) > ELEMENTARY_SIZE; --log_stride) {
        dim3 block(MAX_BLOCK_SIZE / 2);
        dim3 grid(div_up(batch_size, WORK_PER_THREAD), div_up(n / 2, MAX_BLOCK_SIZE / 2), nstack);
        butterfly_multiply_untied_svd_onestep_cuda_kernel<scalar_t, false>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, output_a, log_stride, log_n);
      }
      int stride = 1 << log_stride;
      dim3 block(stride);
      dim3 grid(batch_size, div_up(n / 2, stride), nstack);
      return_intermediates ? butterfly_multiply_untied_svd_cuda_kernel<scalar_t, false, true>
        <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, output_a, log_stride, log_n)
                            : butterfly_multiply_untied_svd_cuda_kernel<scalar_t, false, false>
        <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, output_a, log_stride, log_n);
    }
  });
  AT_CHECK(cudaGetLastError() == cudaSuccess,
     "butterfly_multiply_untied_svd_cuda failed with error code ",
     cudaGetLastError());
}

template <typename scalar_t, typename accscalar_t, bool increasing_stride>
__global__ void butterfly_multiply_untied_svd_backward_cuda_kernel(const at::PackedTensorAccessor<scalar_t, 5> twiddle_a,
                                                                   const at::PackedTensorAccessor<scalar_t, 4> output_a,
                                                                   at::PackedTensorAccessor<scalar_t, 5> d_twiddle_a,
                                                                   at::PackedTensorAccessor<scalar_t, 3> d_input_a,
                                                                   int log_max_stride,
                                                                   int log_n) {
  const int batch_size = output_a.size(1);
  const int s = blockIdx.z;
  const int max_stride = 1 << log_max_stride;
  const int input_base_idx = blockIdx.y * blockDim.x * 2;
  __shared__ scalar_t s_grad[ELEMENTARY_SIZE * 2];
  int b = blockIdx.x * blockDim.y + threadIdx.y;
  if (b < batch_size) {  // Currently we assume 1 batch per thread block, so all threads in the block should enter (otherwise deadlock)
    for (int i = threadIdx.x; i < max_stride * 2; i += blockDim.x) {
      s_grad[i] = d_input_a[b][s][input_base_idx + i];
    }
    int i = threadIdx.x;
    int first_idx = increasing_stride ? 0 : log_n - 1 - log_max_stride;
    for (int idx = first_idx + log_max_stride; idx >= first_idx; --idx) {
      int log_stride = increasing_stride ? idx : log_n - 1 - idx;
      int stride = 1 << log_stride;
      int low_order_bits = i & (stride - 1);  // int low_order_bits = i % stride;
      int pos = 2 * (i - low_order_bits) + low_order_bits;
      const scalar_t twiddle_val[2][2] = {{twiddle_a[s][log_stride][input_base_idx / 2 + i][0][0], twiddle_a[s][log_stride][input_base_idx / 2 + i][0][1]},
                                          {twiddle_a[s][log_stride][input_base_idx / 2 + i][1][0], twiddle_a[s][log_stride][input_base_idx / 2 + i][1][1]}};
      const scalar_t sin_theta = thc_sin(twiddle_val[0][0]), cos_theta = thc_cos(twiddle_val[0][0]);
      const scalar_t sin_phi = thc_sin(twiddle_val[0][1]), cos_phi = thc_cos(twiddle_val[0][1]);
      __syncthreads();
      const scalar_t grad_val[2] = {s_grad[pos], s_grad[pos + stride]};
      scalar_t grad_temp_theta[2];
      thrust::tie(grad_temp_theta[0], grad_temp_theta[1])
        = mult2x2(cos_theta, sin_theta, -sin_theta, cos_theta, grad_val[0], grad_val[1]);
      const scalar_t grad_temp_diag[2] = {grad_temp_theta[0] * twiddle_val[1][0], grad_temp_theta[1] * twiddle_val[1][1]};
      thrust::tie(s_grad[pos], s_grad[pos + stride])
        = mult2x2(cos_phi, sin_phi, -sin_phi, cos_phi, grad_temp_diag[0], grad_temp_diag[1]);
      const scalar_t input_val[2] = {output_a[idx][b][s][input_base_idx + pos], output_a[idx][b][s][input_base_idx + pos + stride]};
      scalar_t input_temp_phi[2];
      thrust::tie(input_temp_phi[0], input_temp_phi[1])
        = mult2x2(cos_phi, -sin_phi, sin_phi, cos_phi, input_val[0], input_val[1]);
      const scalar_t input_temp_diag[2] = {input_temp_phi[0] * twiddle_val[1][0], input_temp_phi[1] * twiddle_val[1][1]};
      accscalar_t d_twiddle_val[2][2];
      // d_theta
      d_twiddle_val[0][0]
        = (grad_val[0] * input_temp_diag[0] + grad_val[1] * input_temp_diag[1]) * (-sin_theta)
        + (-grad_val[0] * input_temp_diag[1] + grad_val[1] * input_temp_diag[0]) * cos_theta;
      // d_sigma_1 and d_sigma_2
      d_twiddle_val[1][0] = grad_temp_theta[0] * input_temp_phi[0];
      d_twiddle_val[1][1] = grad_temp_theta[1] * input_temp_phi[1];
      // d_phi
      d_twiddle_val[0][1]
        = (grad_temp_diag[0] * input_val[0] + grad_temp_diag[1] * input_val[1]) * (-sin_phi)
        + (-grad_temp_diag[0] * input_val[1] + grad_temp_diag[1] * input_val[0]) * cos_phi;
      atomicAdd(&d_twiddle_a[s][log_stride][input_base_idx / 2 + i][0][0], d_twiddle_val[0][0]);
      atomicAdd(&d_twiddle_a[s][log_stride][input_base_idx / 2 + i][0][1], d_twiddle_val[0][1]);
      atomicAdd(&d_twiddle_a[s][log_stride][input_base_idx / 2 + i][1][0], d_twiddle_val[1][0]);
      atomicAdd(&d_twiddle_a[s][log_stride][input_base_idx / 2 + i][1][1], d_twiddle_val[1][1]);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < max_stride * 2; i += blockDim.x) {
      d_input_a[b][s][input_base_idx + i] = s_grad[i];
    }
  }
}

template <typename scalar_t, typename accscalar_t, bool increasing_stride>
__global__ void butterfly_multiply_untied_svd_backward_onestep_cuda_kernel(const at::PackedTensorAccessor<scalar_t, 5> twiddle_a,
                                                                           const at::PackedTensorAccessor<scalar_t, 4> output_a,
                                                                           at::PackedTensorAccessor<scalar_t, 5> d_twiddle_a,
                                                                           at::PackedTensorAccessor<scalar_t, 3> d_input_a,
                                                                           int log_stride,
                                                                           int log_n) {
  const int batch_size = output_a.size(1);
  const int s = blockIdx.z;
  const int idx = increasing_stride ? log_stride : (log_n - 1 - log_stride);  // Index to access output_a
  const int n = output_a.size(3);
  int stride = 1 << log_stride;
  int i = blockIdx.y * blockDim.x + threadIdx.x;
  if (i > n) return;
  int low_order_bits = i & (stride - 1);  // int low_order_bits = i % stride;
  int pos = 2 * (i - low_order_bits) + low_order_bits;
  const scalar_t twiddle_val[2][2] = {{twiddle_a[s][log_stride][i][0][0], twiddle_a[s][log_stride][i][0][1]},
                                      {twiddle_a[s][log_stride][i][1][0], twiddle_a[s][log_stride][i][1][1]}};
  const scalar_t sin_theta = thc_sin(twiddle_val[0][0]), cos_theta = thc_cos(twiddle_val[0][0]);
  const scalar_t sin_phi = thc_sin(twiddle_val[0][1]), cos_phi = thc_cos(twiddle_val[0][1]);
  accscalar_t d_twiddle_val[2][2] = {{0, 0}, {0, 0}};
  for (int b = blockIdx.x * blockDim.y + threadIdx.y; b < batch_size; b += blockDim.y * gridDim.x) {
    const scalar_t grad_val[2] = {d_input_a[b][s][pos], d_input_a[b][s][pos + stride]};
    scalar_t grad_temp_theta[2];
    thrust::tie(grad_temp_theta[0], grad_temp_theta[1])
      = mult2x2(cos_theta, sin_theta, -sin_theta, cos_theta, grad_val[0], grad_val[1]);
    const scalar_t grad_temp_diag[2] = {grad_temp_theta[0] * twiddle_val[1][0], grad_temp_theta[1] * twiddle_val[1][1]};
    thrust::tie(d_input_a[b][s][pos], d_input_a[b][s][pos + stride])
      = mult2x2(cos_phi, sin_phi, -sin_phi, cos_phi, grad_temp_diag[0], grad_temp_diag[1]);
    const scalar_t input_val[2] = {output_a[idx][b][s][pos], output_a[idx][b][s][pos + stride]};
    scalar_t input_temp_phi[2];
    thrust::tie(input_temp_phi[0], input_temp_phi[1])
      = mult2x2(cos_phi, -sin_phi, sin_phi, cos_phi, input_val[0], input_val[1]);
    const scalar_t input_temp_diag[2] = {input_temp_phi[0] * twiddle_val[1][0], input_temp_phi[1] * twiddle_val[1][1]};
    // d_theta
    d_twiddle_val[0][0]
      += (grad_val[0] * input_temp_diag[0] + grad_val[1] * input_temp_diag[1]) * (-sin_theta)
      + (-grad_val[0] * input_temp_diag[1] + grad_val[1] * input_temp_diag[0]) * cos_theta;
    // d_sigma_1 and d_sigma_2
    d_twiddle_val[1][0] += grad_temp_theta[0] * input_temp_phi[0];
    d_twiddle_val[1][1] += grad_temp_theta[1] * input_temp_phi[1];
    // d_phi
    d_twiddle_val[0][1]
      += (grad_temp_diag[0] * input_val[0] + grad_temp_diag[1] * input_val[1]) * (-sin_phi)
      + (-grad_temp_diag[0] * input_val[1] + grad_temp_diag[1] * input_val[0]) * cos_phi;
  }
  atomicAdd(&d_twiddle_a[s][log_stride][i][0][0], d_twiddle_val[0][0]);
  atomicAdd(&d_twiddle_a[s][log_stride][i][0][1], d_twiddle_val[0][1]);
  atomicAdd(&d_twiddle_a[s][log_stride][i][1][0], d_twiddle_val[1][0]);
  atomicAdd(&d_twiddle_a[s][log_stride][i][1][1], d_twiddle_val[1][1]);
}

void butterfly_multiply_untied_svd_backward_cuda(const at::Tensor& twiddle, const at::Tensor& output,
                                                 at::Tensor& d_twiddle, at::Tensor& d_input, bool increasing_stride) {
  const int batch_size = output.size(1);
  const int nstack = output.size(2);
  const int n = output.size(3);
  const int log_n = int(log2((double) n));
  AT_DISPATCH_FLOATING_TYPES(output.scalar_type(), "butterfly_multiply_untied_svd_backward_cuda", [&] {
    using accscalar_t = at::acc_type<scalar_t, true>;
    const auto twiddle_a = twiddle.packed_accessor<scalar_t, 5>();
    const auto output_a = output.packed_accessor<scalar_t, 4>();
    auto d_twiddle_a = d_twiddle.packed_accessor<scalar_t, 5>();
    auto d_input_a = d_input.packed_accessor<scalar_t, 3>();
    if (increasing_stride) {
      int log_stride = log_n - 1;
      for (; (1 << log_stride) > ELEMENTARY_SIZE; --log_stride) {
        dim3 block(MAX_BLOCK_SIZE / 2);
        dim3 grid(div_up(batch_size, WORK_PER_THREAD), div_up(n / 2, MAX_BLOCK_SIZE / 2), nstack);
        butterfly_multiply_untied_svd_backward_onestep_cuda_kernel<scalar_t, accscalar_t, true>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, output_a, d_twiddle_a, d_input_a, log_stride, log_n);
      }
      int stride = 1 << log_stride;
      dim3 block(stride);
      dim3 grid(batch_size, div_up(n / 2, stride), nstack);
      butterfly_multiply_untied_svd_backward_cuda_kernel<scalar_t, accscalar_t, true>
        <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, output_a, d_twiddle_a, d_input_a, log_stride, log_n);
    } else {
      int stride = std::min<int>(ELEMENTARY_SIZE, n / 2);
      int log_stride = int(log2((double) stride));
      dim3 block(stride);
      dim3 grid(batch_size, div_up(n / 2, stride), nstack);
      butterfly_multiply_untied_svd_backward_cuda_kernel<scalar_t, accscalar_t, false>
        <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, output_a, d_twiddle_a, d_input_a, log_stride, log_n);
      for (log_stride++; log_stride <= log_n - 1; ++log_stride) {
        dim3 block(MAX_BLOCK_SIZE / 2);
        dim3 grid(div_up(batch_size, WORK_PER_THREAD), div_up(n / 2, MAX_BLOCK_SIZE / 2), nstack);
        butterfly_multiply_untied_svd_backward_onestep_cuda_kernel<scalar_t, accscalar_t, false>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, output_a, d_twiddle_a, d_input_a, log_stride, log_n);
      }
    }
  });
  AT_CHECK(cudaGetLastError() == cudaSuccess,
     "butterfly_multiply_untied_svd_backward_cuda failed with error code ",
     cudaGetLastError());
}

template <typename scalar_t, typename accscalar_t, bool increasing_stride>
__global__ void butterfly_multiply_untied_svd_forward_backward_cuda_kernel(const CudaAcsr<scalar_t, 5> twiddle_a,
                                                                       const CudaAcsr<scalar_t, 3> input_a,
                                                                       CudaAcsr<scalar_t, 5> d_twiddle_a,
                                                                       CudaAcsr<scalar_t, 3> d_input_a,
                                                                       int log_max_stride,
                                                                       int log_n) {
  const int batch_size = input_a.size(0);
  const int s = blockIdx.z;
  const int max_stride = 1 << log_max_stride;
  const int input_base_idx = blockIdx.y * blockDim.x * 2;
  __shared__ scalar_t s_input[ELEMENTARY_SIZE * 2];
  __shared__ accscalar_t s_twiddle[ELEMENTARY_SIZE][2][2];  // Use accscalar_t instead of scalar_t since we'll reuse the storage for s_d_twiddle
  // Forward pass to compute the intermediate values
  scalar_t input_val_storage[MAX_N_FACTORS][2];  // Storing inputs for backward pass
  int b = blockIdx.x * blockDim.y + threadIdx.y;
  if (b < batch_size) {
    for (int i = threadIdx.x; i < max_stride * 2; i += blockDim.x) {
      s_input[i + threadIdx.y * max_stride * 2] = input_a[b][s][input_base_idx + i];
    }
  }
  int tid_x = threadIdx.x;
  int tid_y = threadIdx.y;
  int first_idx = increasing_stride ? 0 : log_n - 1 - log_max_stride;
  // for (int idx = first_idx; idx < first_idx + log_max_stride; ++idx) {  // Don't need the final output, so skip last step
  for (int idx = first_idx; idx <= first_idx + log_max_stride; ++idx) {  // Let's not skip steps for now
    int log_stride = increasing_stride ? idx : log_n - 1 - idx;
    int stride = 1 << log_stride;
    if (tid_y == 0) {
      s_twiddle[tid_x][0][0] = twiddle_a[s][log_stride][input_base_idx / 2 + tid_x][0][0];
      s_twiddle[tid_x][0][1] = twiddle_a[s][log_stride][input_base_idx / 2 + tid_x][0][1];
      s_twiddle[tid_x][1][0] = twiddle_a[s][log_stride][input_base_idx / 2 + tid_x][1][0];
      s_twiddle[tid_x][1][1] = twiddle_a[s][log_stride][input_base_idx / 2 + tid_x][1][1];
    }
    int low_order_bits = tid_x & (stride - 1);  // int low_order_bits = tid_x % stride;
    int pos_x = 2 * (tid_x - low_order_bits) + low_order_bits;
    int pos_y = tid_y * max_stride * 2;
    int pos = pos_x + pos_y;
    __syncthreads();
    const scalar_t twiddle_val[2][2] = {{s_twiddle[tid_x][0][0], s_twiddle[tid_x][0][1]},
                                        {s_twiddle[tid_x][1][0], s_twiddle[tid_x][1][1]}};
    __syncthreads();  // otherwise some thread might go back to writing to s_twiddle before other thread can read
    if (b < batch_size) {
      input_val_storage[idx - first_idx][0] = s_input[pos];
      input_val_storage[idx - first_idx][1] = s_input[pos + stride];
      const scalar_t sin_theta = thc_sin(twiddle_val[0][0]), cos_theta = thc_cos(twiddle_val[0][0]);
      const scalar_t sin_phi = thc_sin(twiddle_val[0][1]), cos_phi = thc_cos(twiddle_val[0][1]);
      const scalar_t input_val[2] = {s_input[pos], s_input[pos + stride]};
      scalar_t temp[2];
      thrust::tie(temp[0], temp[1]) = mult2x2(cos_phi, -sin_phi, sin_phi, cos_phi, input_val[0], input_val[1]);
      thrust::tie(temp[0], temp[1]) = mult2x2(cos_theta, -sin_theta, sin_theta, cos_theta,
                                              temp[0] * twiddle_val[1][0], temp[1] * twiddle_val[1][1]);
      s_input[pos] = temp[0];
      s_input[pos + stride] = temp[1];
    }
  }
  __syncthreads(); // Otherwise s_input will be overwritten with s_grad before some thread can read
  // Backward pass
  scalar_t* s_grad = &s_input[0]; // Reusing the same storage as s_input
  accscalar_t* s_d_twiddle = (accscalar_t *)&s_twiddle[0][0][0];  // Reusing the same storage as s_twiddle, have to be careful if we change the implemetnation.
  if (b < batch_size) {
    for (int i = threadIdx.x; i < max_stride * 2; i += blockDim.x) {
      s_grad[i + threadIdx.y * max_stride * 2] = d_input_a[b][s][input_base_idx + i];
    }
  }
  for (int idx = first_idx + log_max_stride; idx >= first_idx; --idx) {
    int log_stride = increasing_stride ? idx : log_n - 1 - idx;
    int stride = 1 << log_stride;
    if (tid_y == 0) {
      s_twiddle[tid_x][0][0] = twiddle_a[s][log_stride][input_base_idx / 2 + tid_x][0][0];
      s_twiddle[tid_x][0][1] = twiddle_a[s][log_stride][input_base_idx / 2 + tid_x][0][1];
      s_twiddle[tid_x][1][0] = twiddle_a[s][log_stride][input_base_idx / 2 + tid_x][1][0];
      s_twiddle[tid_x][1][1] = twiddle_a[s][log_stride][input_base_idx / 2 + tid_x][1][1];
    }
    int low_order_bits = tid_x & (stride - 1);  // int low_order_bits = tid_x % stride;
    int pos_x = 2 * (tid_x - low_order_bits) + low_order_bits;
    int pos_y = tid_y * max_stride * 2;
    int pos = pos_x + pos_y;
    __syncthreads();
    const scalar_t twiddle_val[2][2] = {{s_twiddle[tid_x][0][0], s_twiddle[tid_x][0][1]},
                                        {s_twiddle[tid_x][1][0], s_twiddle[tid_x][1][1]}};
    // Don't need to sync here since we sync later at sum_strided_atomic, so no writing to s_twiddle can occur until then
    accscalar_t d_twiddle_val[2][2] = {{0, 0}, {0, 0}};
    if (b < batch_size) {
      const scalar_t sin_theta = thc_sin(twiddle_val[0][0]), cos_theta = thc_cos(twiddle_val[0][0]);
      const scalar_t sin_phi = thc_sin(twiddle_val[0][1]), cos_phi = thc_cos(twiddle_val[0][1]);
      const scalar_t grad_val[2] = {s_grad[pos], s_grad[pos + stride]};
      scalar_t grad_temp_theta[2];
      thrust::tie(grad_temp_theta[0], grad_temp_theta[1])
        = mult2x2(cos_theta, sin_theta, -sin_theta, cos_theta, grad_val[0], grad_val[1]);
      const scalar_t grad_temp_diag[2] = {grad_temp_theta[0] * twiddle_val[1][0], grad_temp_theta[1] * twiddle_val[1][1]};
      thrust::tie(s_grad[pos], s_grad[pos + stride])
        = mult2x2(cos_phi, sin_phi, -sin_phi, cos_phi, grad_temp_diag[0], grad_temp_diag[1]);
      const scalar_t input_val[2] = {input_val_storage[idx - first_idx][0], input_val_storage[idx - first_idx][1]};
      scalar_t input_temp_phi[2];
      thrust::tie(input_temp_phi[0], input_temp_phi[1])
        = mult2x2(cos_phi, -sin_phi, sin_phi, cos_phi, input_val[0], input_val[1]);
      const scalar_t input_temp_diag[2] = {input_temp_phi[0] * twiddle_val[1][0], input_temp_phi[1] * twiddle_val[1][1]};
      // d_theta
      d_twiddle_val[0][0]
        = (grad_val[0] * input_temp_diag[0] + grad_val[1] * input_temp_diag[1]) * (-sin_theta)
        + (-grad_val[0] * input_temp_diag[1] + grad_val[1] * input_temp_diag[0]) * cos_theta;
      // d_sigma_1 and d_sigma_2
      d_twiddle_val[1][0] = grad_temp_theta[0] * input_temp_phi[0];
      d_twiddle_val[1][1] = grad_temp_theta[1] * input_temp_phi[1];
      // d_phi
      d_twiddle_val[0][1]
        = (grad_temp_diag[0] * input_val[0] + grad_temp_diag[1] * input_val[1]) * (-sin_phi)
        + (-grad_temp_diag[0] * input_val[1] + grad_temp_diag[1] * input_val[0]) * cos_phi;
    }
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int nthreads = blockDim.x * blockDim.y;
    sum_strided_atomic(reinterpret_cast<accscalar_t (&)[4]>(d_twiddle_val), s_d_twiddle, max_stride, nthreads, tid);
    if (tid_y == 0) {
      atomicAdd(&d_twiddle_a[s][log_stride][input_base_idx / 2 + tid_x][0][0], s_d_twiddle[tid_x]);
      atomicAdd(&d_twiddle_a[s][log_stride][input_base_idx / 2 + tid_x][0][1], s_d_twiddle[tid_x + max_stride]);
      atomicAdd(&d_twiddle_a[s][log_stride][input_base_idx / 2 + tid_x][1][0], s_d_twiddle[tid_x + 2 * max_stride]);
      atomicAdd(&d_twiddle_a[s][log_stride][input_base_idx / 2 + tid_x][1][1], s_d_twiddle[tid_x + 3 * max_stride]);
    }
    __syncthreads();  // Otherwise s_d_twiddle will be overwritten with s_twiddle before some thread can read
  }
  if (b < batch_size) {
    for (int i = threadIdx.x; i < max_stride * 2; i += blockDim.x) {
      d_input_a[b][s][input_base_idx + i] = s_grad[i + threadIdx.y * max_stride * 2];
    }
  }
}

void butterfly_multiply_untied_svd_forward_backward_cuda(const at::Tensor& twiddle, const at::Tensor& input,
                                                         at::Tensor& d_twiddle, at::Tensor& d_input, bool increasing_stride) {
  const int batch_size = input.size(0);
  const int nstack = input.size(1);
  const int n = input.size(2);
  const int log_n = int(log2((double) n));
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "butterfly_multiply_untied_svd_forward_backward_cuda", [&] {
    using accscalar_t = at::acc_type<scalar_t, true>;
    const auto twiddle_a = twiddle.packed_accessor<scalar_t, 5, at::RestrictPtrTraits, int32_t>();
    const auto input_a = input.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>();
    auto d_twiddle_a = d_twiddle.packed_accessor<scalar_t, 5, at::RestrictPtrTraits, int32_t>();
    auto d_input_a = d_input.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>();
    int stride = std::min<int>(ELEMENTARY_SIZE, n / 2);
    int log_stride = int(log2((double) stride));
    dim3 block(stride, div_up(MAX_BLOCK_SIZE, stride * 2));
    dim3 grid(div_up(batch_size, block.y), div_up(n / 2, stride), nstack);
    increasing_stride ? butterfly_multiply_untied_svd_forward_backward_cuda_kernel<scalar_t, accscalar_t, true>
      <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, input_a, d_twiddle_a, d_input_a, log_stride, log_n)
                      : butterfly_multiply_untied_svd_forward_backward_cuda_kernel<scalar_t, accscalar_t, false>
      <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, input_a, d_twiddle_a, d_input_a, log_stride, log_n);
  });
  AT_CHECK(cudaGetLastError() == cudaSuccess,
     "butterfly_multiply_untied_svd_forward_backward_cuda failed with error code ",
     cudaGetLastError());
}

template <typename scalar_t, bool increasing_stride, bool return_intermediates>
__global__ void butterfly_conv2d_svd_cuda_kernel(const at::PackedTensorAccessor<scalar_t, 5> twiddle_a,
                                                 const at::PackedTensorAccessor<scalar_t, 4> input_a,
                                                 at::PackedTensorAccessor<scalar_t, 4> output_a,
                                                 int log_max_stride,
                                                 int log_n,
                                                 int kernel_size,
                                                 int padding,
                                                 int h_out,
                                                 int w_out) {
  const int batch_size = output_a.size(1);
  const int stack = blockIdx.z;
  const int s = blockIdx.y + gridDim.y * stack;
  const int max_stride = 1 << log_max_stride;
  // base index always 0
  const int input_base_idx = 0;
  const int h_in = input_a.size(2);
  const int w_in = input_a.size(3);
  __shared__ scalar_t s_input[ELEMENTARY_SIZE * 2];
  __shared__ scalar_t s_twiddle[ELEMENTARY_SIZE][2][2];
  int b = blockIdx.x * blockDim.y + threadIdx.y;
  const int patch_idx = b % (h_out * w_out);
  const int batch_idx = b / (h_out * w_out);
  int first_idx = increasing_stride ? 0 : log_n - 1 - log_max_stride;
  if (b < batch_size) {
    for (int t = threadIdx.x; t < max_stride * 2; t += blockDim.x) {
      // get index into patch
      int k_i = stack / kernel_size;
      int k_j = stack % kernel_size;
      // get patch index into full matrix
      int p_i = (patch_idx) / w_out;
      int p_j = (patch_idx) % (w_out);
      // combine indices and adjust for padding
      int i = k_i + p_i - padding;
      int j = k_j + p_j - padding;
      if (i >= w_in or j >= h_in or i < 0 or j < 0) s_input[t + threadIdx.y * max_stride * 2] = 0;
      else{
        s_input[t + threadIdx.y * max_stride * 2] = input_a[batch_idx][input_base_idx + t][i][j];
        // load input into first idx of output for backward pass
        // we allocated this memory already so shouldn't affect too much
        output_a[0][b][s][input_base_idx + t] = s_input[t + threadIdx.y * max_stride * 2];
      }
    }
  }
  int tid_x = threadIdx.x;
  int tid_y = threadIdx.y;
  for (int idx = first_idx; idx <= first_idx + log_max_stride; ++idx) {
    int log_stride = increasing_stride ? idx : log_n - 1 - idx;
    int stride = 1 << log_stride;
    if (tid_y == 0) {
      s_twiddle[tid_x][0][0] = twiddle_a[s][log_stride][input_base_idx / 2 + tid_x][0][0];
      s_twiddle[tid_x][0][1] = twiddle_a[s][log_stride][input_base_idx / 2 + tid_x][0][1];
      s_twiddle[tid_x][1][0] = twiddle_a[s][log_stride][input_base_idx / 2 + tid_x][1][0];
      s_twiddle[tid_x][1][1] = twiddle_a[s][log_stride][input_base_idx / 2 + tid_x][1][1];
    }
    int low_order_bits = tid_x & (stride - 1);  // int low_order_bits = tid_x % stride;
    int pos_x = 2 * (tid_x - low_order_bits) + low_order_bits;
    int pos_y = tid_y * max_stride * 2;
    int pos = pos_x + pos_y;
    __syncthreads();
    const scalar_t twiddle_val[2][2] = {{s_twiddle[tid_x][0][0], s_twiddle[tid_x][0][1]},
                                        {s_twiddle[tid_x][1][0], s_twiddle[tid_x][1][1]}};
    __syncthreads();  // otherwise some thread might go back to writing to s_twiddle before other thread can read
    if (b < batch_size) {
      const scalar_t sin_theta = thc_sin(twiddle_val[0][0]), cos_theta = thc_cos(twiddle_val[0][0]);
      const scalar_t sin_phi = thc_sin(twiddle_val[0][1]), cos_phi = thc_cos(twiddle_val[0][1]);
      const scalar_t input_val[2] = {s_input[pos], s_input[pos + stride]};
      scalar_t temp[2];
      thrust::tie(temp[0], temp[1]) = mult2x2(cos_phi, -sin_phi, sin_phi, cos_phi, input_val[0], input_val[1]);
      thrust::tie(temp[0], temp[1]) = mult2x2(cos_theta, -sin_theta, sin_theta, cos_theta,
                                              temp[0] * twiddle_val[1][0], temp[1] * twiddle_val[1][1]);
      s_input[pos] = temp[0];
      s_input[pos + stride] = temp[1];
      if (return_intermediates || idx == first_idx + log_max_stride) {
        output_a[idx+1][b][s][input_base_idx + pos_x] = s_input[pos];
        output_a[idx+1][b][s][input_base_idx + pos_x + stride] = s_input[pos + stride];
      }
    }
  }
}

void butterfly_conv2d_svd_cuda(const at::Tensor& twiddle,
    const at::Tensor& input, at::Tensor& output,
    const int kernel_size, const int padding,
    const int h_out, const int w_out, bool increasing_stride,
    bool return_intermediates)
{
  const int b_in = input.size(0);
  const int n = input.size(1); /*c*/
  const int nstack = twiddle.size(0);
  const int stack = kernel_size*kernel_size;
  const int log_n = int(log2((double) n));
  const int batch_size = output.size(1);
  AT_DISPATCH_FLOATING_TYPES(output.scalar_type(),
    "butterfly_conv2d_svd_cuda", [&] {
      const auto twiddle_a = twiddle.packed_accessor<scalar_t, 5>();
      // batch_size, c, h, w
      const auto input_a = input.packed_accessor<scalar_t, 4>();
      // log c_in, h*w*batch_size, nstack, c_in
      auto output_a = output.packed_accessor<scalar_t, 4>();
      // assume in_channels <= 1024
      int stride = std::min<int>(ELEMENTARY_SIZE, n / 2);
      int log_stride = int(log2((double) stride));
      // to support out_channels > in_channels
      int c_out_ratio = nstack / stack;
      // dim3 block(stride);
      // dim3 grid(batch_size, c_out_ratio, stack);
      dim3 block(stride, div_up(MAX_BLOCK_SIZE, stride * 2));
      dim3 grid(div_up(batch_size, block.y), c_out_ratio, stack);
      if (increasing_stride) {
        return_intermediates ? butterfly_conv2d_svd_cuda_kernel<scalar_t, true, true>
        <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, input_a,
          output_a, log_stride, log_n, kernel_size, padding, h_out, w_out)
                           : butterfly_conv2d_svd_cuda_kernel<scalar_t, true, false>
        <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, input_a,
          output_a, log_stride, log_n, kernel_size, padding, h_out, w_out);
      }
      else {
        return_intermediates ? butterfly_conv2d_svd_cuda_kernel<scalar_t, false, true>
        <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, input_a,
          output_a, log_stride, log_n, kernel_size, padding, h_out, w_out)
                           : butterfly_conv2d_svd_cuda_kernel<scalar_t, false, false>
        <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, input_a,
          output_a, log_stride, log_n, kernel_size, padding, h_out, w_out);
      }
  });
  AT_CHECK(cudaGetLastError() == cudaSuccess,
     "butterfly_conv2d_svd_cuda failed with error code ",
     cudaGetLastError());
}

template <typename scalar_t, typename accscalar_t, bool increasing_stride>
__global__ void butterfly_conv2d_svd_forward_backward_cuda_kernel(const at::PackedTensorAccessor<scalar_t, 5> twiddle_a,
                                                                  const at::PackedTensorAccessor<scalar_t, 4> input_a,
                                                                  const at::PackedTensorAccessor<scalar_t, 3> grad_a,
                                                                  at::PackedTensorAccessor<scalar_t, 5> d_twiddle_a,
                                                                  at::PackedTensorAccessor<scalar_t, 4> d_input_a,
                                                                  int log_max_stride,
                                                                  int log_n,
                                                                  int kernel_size,
                                                                  int padding,
                                                                  int h_out,
                                                                  int w_out) {
  const int b_out = grad_a.size(0); // b_in * h_out * w_out
  const int h_in = input_a.size(2);
  const int w_in = input_a.size(3);
  const int stack = blockIdx.z;
  const int s = blockIdx.x + gridDim.x * stack;
  const int max_stride = 1 << log_max_stride;
  const int input_base_idx = 0;
  __shared__ scalar_t s_input[ELEMENTARY_SIZE * 2];
  __shared__ accscalar_t s_twiddle[ELEMENTARY_SIZE][2][2];  // Use accscalar_t instead of scalar_t since we'll reuse the storage for s_d_twiddle
  // Forward pass to compute the intermediate values
  scalar_t input_val_storage[MAX_N_FACTORS][2];  // Storing inputs for backward pass
  int b = blockIdx.y * blockDim.y + threadIdx.y;
  const int patch_idx = b % (h_out * w_out);
  const int batch_idx = b / (h_out * w_out);
  if (b < b_out) {
    for (int t = threadIdx.x; t < max_stride * 2; t += blockDim.x) {
      // get index into patch
      int k_i = stack / kernel_size;
      int k_j = stack % kernel_size;
      // get patch index into full matrix
      int p_i = (patch_idx) / w_out;
      int p_j = (patch_idx) % (w_out);
      // combine indices and adjust for padding
      int i = k_i + p_i - padding;
      int j = k_j + p_j - padding;
      if (i >= w_in or j >= h_in or i < 0 or j < 0) s_input[t + threadIdx.y * max_stride * 2] = 0;
      else{
        s_input[t + threadIdx.y * max_stride * 2] = input_a[batch_idx][input_base_idx + t][i][j];
      }
    }
  }
  int tid_x = threadIdx.x;
  int tid_y = threadIdx.y;
  int first_idx = increasing_stride ? 0 : log_n - 1 - log_max_stride;
  for (int idx = first_idx; idx <= first_idx + log_max_stride; ++idx) {  // Let's not skip steps for now
    int log_stride = increasing_stride ? idx : log_n - 1 - idx;
    int stride = 1 << log_stride;
    if (tid_y == 0) {
      s_twiddle[tid_x][0][0] = twiddle_a[s][log_stride][input_base_idx / 2 + tid_x][0][0];
      s_twiddle[tid_x][0][1] = twiddle_a[s][log_stride][input_base_idx / 2 + tid_x][0][1];
      s_twiddle[tid_x][1][0] = twiddle_a[s][log_stride][input_base_idx / 2 + tid_x][1][0];
      s_twiddle[tid_x][1][1] = twiddle_a[s][log_stride][input_base_idx / 2 + tid_x][1][1];
    }
    int low_order_bits = tid_x & (stride - 1);  // int low_order_bits = tid_x % stride;
    int pos_x = 2 * (tid_x - low_order_bits) + low_order_bits;
    int pos_y = tid_y * max_stride * 2;
    int pos = pos_x + pos_y;
    __syncthreads();
    const scalar_t twiddle_val[2][2] = {{s_twiddle[tid_x][0][0], s_twiddle[tid_x][0][1]},
                                        {s_twiddle[tid_x][1][0], s_twiddle[tid_x][1][1]}};
    __syncthreads();  // otherwise some thread might go back to writing to s_twiddle before other thread can read
    if (b < b_out) {
      input_val_storage[idx - first_idx][0] = s_input[pos];
      input_val_storage[idx - first_idx][1] = s_input[pos + stride];
      const scalar_t sin_theta = thc_sin(twiddle_val[0][0]), cos_theta = thc_cos(twiddle_val[0][0]);
      const scalar_t sin_phi = thc_sin(twiddle_val[0][1]), cos_phi = thc_cos(twiddle_val[0][1]);
      const scalar_t input_val[2] = {s_input[pos], s_input[pos + stride]};
      scalar_t temp[2];
      thrust::tie(temp[0], temp[1]) = mult2x2(cos_phi, -sin_phi, sin_phi, cos_phi, input_val[0], input_val[1]);
      thrust::tie(temp[0], temp[1]) = mult2x2(cos_theta, -sin_theta, sin_theta, cos_theta,
                                              temp[0] * twiddle_val[1][0], temp[1] * twiddle_val[1][1]);
      s_input[pos] = temp[0];
      s_input[pos + stride] = temp[1];
    }
  }
  __syncthreads(); // Otherwise s_input will be overwritten with s_grad before some thread can read
  // Backward pass
  scalar_t* s_grad = &s_input[0]; // Reusing the same storage as s_input
  accscalar_t* s_d_twiddle = (accscalar_t *)&s_twiddle[0][0][0];  // Reusing the same storage as s_twiddle, have to be careful if we change the implemetnation.
  if (b < b_out) {
    for (int i = threadIdx.x; i < max_stride * 2; i += blockDim.x) {
      s_grad[i + threadIdx.y * max_stride * 2] = grad_a[b][s][input_base_idx + i];
      if (s_grad[i + threadIdx.y * max_stride * 2] != 0){
      }
    }
  }
  for (int idx = first_idx + log_max_stride; idx >= first_idx; --idx) {
    int log_stride = increasing_stride ? idx : log_n - 1 - idx;
    int stride = 1 << log_stride;
    if (tid_y == 0) {
      s_twiddle[tid_x][0][0] = twiddle_a[s][log_stride][input_base_idx / 2 + tid_x][0][0];
      s_twiddle[tid_x][0][1] = twiddle_a[s][log_stride][input_base_idx / 2 + tid_x][0][1];
      s_twiddle[tid_x][1][0] = twiddle_a[s][log_stride][input_base_idx / 2 + tid_x][1][0];
      s_twiddle[tid_x][1][1] = twiddle_a[s][log_stride][input_base_idx / 2 + tid_x][1][1];
    }
    int low_order_bits = tid_x & (stride - 1);  // int low_order_bits = tid_x % stride;
    int pos_x = 2 * (tid_x - low_order_bits) + low_order_bits;
    int pos_y = tid_y * max_stride * 2;
    int pos = pos_x + pos_y;
    __syncthreads();
    const scalar_t twiddle_val[2][2] = {{s_twiddle[tid_x][0][0], s_twiddle[tid_x][0][1]},
                                        {s_twiddle[tid_x][1][0], s_twiddle[tid_x][1][1]}};
    // Don't need to sync here since we sync later at sum_strided_atomic, so no writing to s_twiddle can occur until then
    accscalar_t d_twiddle_val[2][2] = {{0, 0}, {0, 0}};
    if (b < b_out) {
      const scalar_t sin_theta = thc_sin(twiddle_val[0][0]), cos_theta = thc_cos(twiddle_val[0][0]);
      const scalar_t sin_phi = thc_sin(twiddle_val[0][1]), cos_phi = thc_cos(twiddle_val[0][1]);
      const scalar_t grad_val[2] = {s_grad[pos], s_grad[pos + stride]};
      scalar_t grad_temp_theta[2];
      thrust::tie(grad_temp_theta[0], grad_temp_theta[1])
        = mult2x2(cos_theta, sin_theta, -sin_theta, cos_theta, grad_val[0], grad_val[1]);
      const scalar_t grad_temp_diag[2] = {grad_temp_theta[0] * twiddle_val[1][0], grad_temp_theta[1] * twiddle_val[1][1]};
      thrust::tie(s_grad[pos], s_grad[pos + stride])
        = mult2x2(cos_phi, sin_phi, -sin_phi, cos_phi, grad_temp_diag[0], grad_temp_diag[1]);
      const scalar_t input_val[2] = {input_val_storage[idx - first_idx][0], input_val_storage[idx - first_idx][1]};
      scalar_t input_temp_phi[2];
      thrust::tie(input_temp_phi[0], input_temp_phi[1])
        = mult2x2(cos_phi, -sin_phi, sin_phi, cos_phi, input_val[0], input_val[1]);
      const scalar_t input_temp_diag[2] = {input_temp_phi[0] * twiddle_val[1][0], input_temp_phi[1] * twiddle_val[1][1]};
      // d_theta
      d_twiddle_val[0][0]
        = (grad_val[0] * input_temp_diag[0] + grad_val[1] * input_temp_diag[1]) * (-sin_theta)
        + (-grad_val[0] * input_temp_diag[1] + grad_val[1] * input_temp_diag[0]) * cos_theta;
      // d_sigma_1 and d_sigma_2
      d_twiddle_val[1][0] = grad_temp_theta[0] * input_temp_phi[0];
      d_twiddle_val[1][1] = grad_temp_theta[1] * input_temp_phi[1];
      // d_phi
      d_twiddle_val[0][1]
        = (grad_temp_diag[0] * input_val[0] + grad_temp_diag[1] * input_val[1]) * (-sin_phi)
        + (-grad_temp_diag[0] * input_val[1] + grad_temp_diag[1] * input_val[0]) * cos_phi;
    }
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int nthreads = blockDim.x * blockDim.y;
    sum_strided_atomic(reinterpret_cast<accscalar_t (&)[4]>(d_twiddle_val), s_d_twiddle, max_stride, nthreads, tid);
    if (tid_y == 0) {
      atomicAdd(&d_twiddle_a[s][log_stride][input_base_idx / 2 + tid_x][0][0], s_d_twiddle[tid_x]);
      atomicAdd(&d_twiddle_a[s][log_stride][input_base_idx / 2 + tid_x][0][1], s_d_twiddle[tid_x + max_stride]);
      atomicAdd(&d_twiddle_a[s][log_stride][input_base_idx / 2 + tid_x][1][0], s_d_twiddle[tid_x + 2 * max_stride]);
      atomicAdd(&d_twiddle_a[s][log_stride][input_base_idx / 2 + tid_x][1][1], s_d_twiddle[tid_x + 3 * max_stride]);
    }
    __syncthreads();  // Otherwise s_d_twiddle will be overwritten with s_twiddle before some thread can read
  }
  if (b < b_out) {
    for (int t = threadIdx.x; t < max_stride * 2; t += blockDim.x) {
      // map back to b, c, h, w
      // get index into patch
      int k_i = stack / kernel_size; // s / kernel_size
      int k_j = stack % kernel_size; // stack % kernel_size
      // get patch index into full matrix
      int p_i = (patch_idx) / w_out;
      int p_j = (patch_idx) % (w_out);
      // combine indices and adjust for padding
      int i = k_i + p_i - padding;
      int j = k_j + p_j - padding;
      // this needs to be atomic because input is reused in forward pass
      // with out_channels > in_channels and for each entry of the patch
      if (i < w_in && j < h_in && i >= 0 && j >= 0) {
        atomicAdd(&d_input_a[batch_idx][input_base_idx + t][i][j], s_grad[t + threadIdx.y * max_stride * 2]);
      }
    }
  }
}

void butterfly_conv2d_svd_forward_backward_cuda(const at::Tensor& twiddle,
    const at::Tensor& input, const at::Tensor&grad,
    at::Tensor& d_twiddle, at::Tensor& d_input,
    const int kernel_size, const int padding, const int h_out, const int w_out,
    bool increasing_stride) {
  const int batch_size = grad.size(0); // b_out = b_in * h_out * w_out
  const int nstack = twiddle.size(0);
  const int stack = kernel_size * kernel_size;
  const int n = d_input.size(1); // c_in
  const int log_n = int(log2((double) n));
  const int c_out_ratio = nstack / stack;
  AT_DISPATCH_FLOATING_TYPES(grad.scalar_type(),
  "butterfly_conv2d_svd_forward_backward_cuda", [&] {
    using accscalar_t = at::acc_type<scalar_t, true>;
    const auto twiddle_a = twiddle.packed_accessor<scalar_t, 5>();
    const auto input_a = input.packed_accessor<scalar_t, 4>();
    const auto grad_a = grad.packed_accessor<scalar_t, 3>();
    auto d_twiddle_a = d_twiddle.packed_accessor<scalar_t, 5>();
    auto d_input_a = d_input.packed_accessor<scalar_t, 4>();
    int stride = std::min<int>(ELEMENTARY_SIZE, n / 2);
    int log_stride = int(log2((double) stride));
    dim3 block(stride, div_up(MAX_BLOCK_SIZE, stride * 2));
    dim3 grid(c_out_ratio, div_up(batch_size, block.y), stack);
    increasing_stride ?
      butterfly_conv2d_svd_forward_backward_cuda_kernel<scalar_t, accscalar_t, true>
      <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        twiddle_a, input_a, grad_a, d_twiddle_a, d_input_a, log_stride,
        log_n, kernel_size, padding, h_out, w_out) :
      butterfly_conv2d_svd_forward_backward_cuda_kernel<scalar_t, accscalar_t, false>
        <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        twiddle_a, input_a, grad_a, d_twiddle_a, d_input_a, log_stride,
        log_n, kernel_size, padding, h_out, w_out);
  });
  AT_CHECK(cudaGetLastError() == cudaSuccess,
     "butterfly_conv2d_svd_backward_cuda failed with error code ",
     cudaGetLastError());
}


template <typename scalar_t>
__global__ void permutation_factor_even_odd_multiply_cuda_kernel(const at::PackedTensorAccessor<scalar_t, 1> p_a,
                                                                 const at::PackedTensorAccessor<scalar_t, 3> input_a,
                                                                 const at::PackedTensorAccessor<scalar_t, 3> permuted_input_a,
                                                                 at::PackedTensorAccessor<scalar_t, 3> output_a) {
  const auto p = p_a[0];
  const auto batch_size = input_a.size(0);
  const auto n = input_a.size(2);  // already divided by 2
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    for (int64_t b = blockIdx.y * blockDim.y + threadIdx.y; b < batch_size; b += blockDim.y * gridDim.y) {
      #pragma unroll
      for (int j = 0; j <= 1; ++j) {
        output_a[b][j][i] = (1 - p) * input_a[b][j][i] + p * permuted_input_a[b][j][i];
      }
    }
  }
}

template <typename scalar_t>
__global__ void permutation_factor_even_odd_multiply_complex_cuda_kernel(const at::PackedTensorAccessor<scalar_t, 1> p_a,
                                                                         const at::PackedTensorAccessor<scalar_t, 4> input_a,
                                                                         const at::PackedTensorAccessor<scalar_t, 4> permuted_input_a,
                                                                         at::PackedTensorAccessor<scalar_t, 4> output_a) {
  const auto p = p_a[0];
  const auto batch_size = input_a.size(0);
  const auto n = input_a.size(2);
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    for (int64_t b = blockIdx.y * blockDim.y + threadIdx.y; b < batch_size; b += blockDim.y * gridDim.y) {
      #pragma unroll
      for (int j = 0; j <= 1; ++j) {
        #pragma unroll
        for (int k = 0; k <= 1; ++k) {
          output_a[b][j][i][k] = (1 - p) * input_a[b][j][i][k] + p * permuted_input_a[b][j][i][k];
        }
      }
    }
  }
}

void permutation_factor_even_odd_multiply_cuda(const at::Tensor& p, const at::Tensor& input, at::Tensor& output) {
  const auto batch_size = input.size(0);
  const auto n = input.size(1);
  dim3 block;
  block.x = std::min<int64_t>(MAX_BLOCK_SIZE, n / 2);
  block.y = div_up(MAX_BLOCK_SIZE, block.x);
  dim3 grid(div_up(n / 2, block.x), div_up(batch_size, block.y * WORK_PER_THREAD));
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "permutation_factor_even_odd_multiply", [&] {
    const auto p_a = p.packed_accessor<scalar_t, 1>();
    switch (input.dim()) {
      case 2: // real
        {
          const auto permuted_input = input.reshape({batch_size, n / 2, 2}).transpose(1, 2);
          const auto input_folded = input.reshape({batch_size, 2, n / 2});
          output = output.view({batch_size, 2, n / 2});
          const auto input_a = input_folded.packed_accessor<scalar_t, 3>();
          const auto permuted_input_a = permuted_input.packed_accessor<scalar_t, 3>();
          auto output_a = output.packed_accessor<scalar_t, 3>();
          permutation_factor_even_odd_multiply_cuda_kernel<scalar_t>
            <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(p_a, input_a, permuted_input_a, output_a);
          output = output.view({batch_size, n});
          break;
        }
      case 3: // complex
        {
          const auto permuted_input = input.reshape({batch_size, n / 2, 2, 2}).transpose(1, 2);
          const auto input_folded = input.reshape({batch_size, 2, n / 2, 2});
          output = output.view({batch_size, 2, n / 2, 2});
          const auto input_a = input_folded.packed_accessor<scalar_t, 4>();
          const auto permuted_input_a = permuted_input.packed_accessor<scalar_t, 4>();
          auto output_a = output.packed_accessor<scalar_t, 4>();
          permutation_factor_even_odd_multiply_complex_cuda_kernel<scalar_t>
            <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(p_a, input_a, permuted_input_a, output_a);
          output = output.view({batch_size, n, 2});
          break;
        }
      default:
        AT_ERROR("permutation_factor_even_odd_multiply requires input dimension 2 or 3");
    }
  });
  AT_CHECK(cudaGetLastError() == cudaSuccess,
     "permutation_factor_even_odd_multiply_cuda failed with error code ",
     cudaGetLastError());
}

template <typename scalar_t>
__global__ void permutation_factor_even_odd_multiply_backward_cuda_kernel(const at::PackedTensorAccessor<scalar_t, 3> grad_a,
                                                                          const at::PackedTensorAccessor<scalar_t, 3> grad_reshaped_a,
                                                                          const at::PackedTensorAccessor<scalar_t, 3> permuted_grad_a,
                                                                          const at::PackedTensorAccessor<scalar_t, 1> p_a,
                                                                          const at::PackedTensorAccessor<scalar_t, 3> input_a,
                                                                          const at::PackedTensorAccessor<scalar_t, 3> permuted_input_a,
                                                                          at::PackedTensorAccessor<scalar_t, 2> d_p_expanded_a,
                                                                          at::PackedTensorAccessor<scalar_t, 3> d_input_a) {
  const scalar_t p = p_a[0];
  const auto batch_size = input_a.size(0);
  const auto n = input_a.size(2);
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    for (int64_t b = blockIdx.y * blockDim.y + threadIdx.y; b < batch_size; b += blockDim.y * gridDim.y) {
      d_p_expanded_a[b][i] = (permuted_input_a[b][0][i] - input_a[b][0][i]) * grad_reshaped_a[b][0][i]
        + (permuted_input_a[b][1][i] - input_a[b][1][i]) * grad_reshaped_a[b][1][i];
      d_input_a[b][i][0] = (1 - p) * grad_a[b][i][0] + p * permuted_grad_a[b][i][0];
      d_input_a[b][i][1] = (1 - p) * grad_a[b][i][1] + p * permuted_grad_a[b][i][1];
    }
  }
}

template <typename scalar_t>
__global__ void permutation_factor_even_odd_multiply_complex_backward_cuda_kernel(const at::PackedTensorAccessor<scalar_t, 4> grad_a,
                                                                                  const at::PackedTensorAccessor<scalar_t, 4> grad_reshaped_a,
                                                                                  const at::PackedTensorAccessor<scalar_t, 4> permuted_grad_a,
                                                                                  const at::PackedTensorAccessor<scalar_t, 1> p_a,
                                                                                  const at::PackedTensorAccessor<scalar_t, 4> input_a,
                                                                                  const at::PackedTensorAccessor<scalar_t, 4> permuted_input_a,
                                                                                  at::PackedTensorAccessor<scalar_t, 2> d_p_expanded_a,
                                                                                  at::PackedTensorAccessor<scalar_t, 4> d_input_a) {
  const scalar_t p = p_a[0];
  const auto batch_size = input_a.size(0);
  const auto n = input_a.size(2);
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    for (int64_t b = blockIdx.y * blockDim.y + threadIdx.y; b < batch_size; b += blockDim.y * gridDim.y) {
      d_p_expanded_a[b][i] = (permuted_input_a[b][0][i][0] - input_a[b][0][i][0]) * grad_reshaped_a[b][0][i][0]
        + (permuted_input_a[b][0][i][1] - input_a[b][0][i][1]) * grad_reshaped_a[b][0][i][1]
        + (permuted_input_a[b][1][i][0] - input_a[b][1][i][0]) * grad_reshaped_a[b][1][i][0]
        + (permuted_input_a[b][1][i][1] - input_a[b][1][i][1]) * grad_reshaped_a[b][1][i][1];
      d_input_a[b][i][0][0] = (1 - p) * grad_a[b][i][0][0] + p * permuted_grad_a[b][i][0][0];
      d_input_a[b][i][0][1] = (1 - p) * grad_a[b][i][0][1] + p * permuted_grad_a[b][i][0][1];
      d_input_a[b][i][1][0] = (1 - p) * grad_a[b][i][1][0] + p * permuted_grad_a[b][i][1][0];
      d_input_a[b][i][1][1] = (1 - p) * grad_a[b][i][1][1] + p * permuted_grad_a[b][i][1][1];
    }
  }
}

void permutation_factor_even_odd_multiply_backward_cuda(const at::Tensor& grad, const at::Tensor& p, const at::Tensor& input,
                                                        at::Tensor& d_p_expanded, at::Tensor& d_input) {
  const auto batch_size = input.size(0);
  const auto n = input.size(1);
  dim3 block;
  block.x = std::min<int64_t>(MAX_BLOCK_SIZE, n / 2);
  block.y = div_up(MAX_BLOCK_SIZE, block.x);
  dim3 grid(div_up(n / 2, block.x), div_up(batch_size, block.y * WORK_PER_THREAD));
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "permutation_factor_even_odd_multiply_backward", [&] {
    const auto p_a = p.packed_accessor<scalar_t, 1>();
    auto d_p_expanded_a = d_p_expanded.packed_accessor<scalar_t, 2>();
    switch (input.dim()) {
      case 2: // real
        {
          const auto permuted_input = input.reshape({batch_size, n / 2, 2}).transpose(1, 2);
          const auto input_folded = input.reshape({batch_size, 2, n / 2});
          const auto grad_reshaped = grad.reshape({batch_size, 2, n / 2});
          const auto permuted_grad = grad.reshape({batch_size, 2, n / 2}).transpose(1, 2);
          const auto grad_folded = grad.reshape({batch_size, n / 2, 2});
          d_input = d_input.view({batch_size, n/ 2, 2});
          // Accessors
          const auto input_a = input_folded.packed_accessor<scalar_t, 3>();
          const auto permuted_input_a = permuted_input.packed_accessor<scalar_t, 3>();
          const auto grad_reshaped_a = grad_reshaped.packed_accessor<scalar_t, 3>();
          const auto grad_a = grad_folded.packed_accessor<scalar_t, 3>();
          const auto permuted_grad_a = permuted_grad.packed_accessor<scalar_t, 3>();
          auto d_input_a = d_input.packed_accessor<scalar_t, 3>();
          permutation_factor_even_odd_multiply_backward_cuda_kernel<scalar_t>
            <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(grad_a, grad_reshaped_a, permuted_grad_a, p_a, input_a, permuted_input_a, d_p_expanded_a, d_input_a);
          d_input = d_input.view({batch_size, n});
          break;
        }
      case 3: // complex
        {
          const auto permuted_input = input.reshape({batch_size, n / 2, 2, 2}).transpose(1, 2);
          const auto input_folded = input.reshape({batch_size, 2, n / 2, 2});
          const auto grad_reshaped = grad.reshape({batch_size, 2, n / 2, 2});
          const auto permuted_grad = grad.reshape({batch_size, 2, n / 2, 2}).transpose(1, 2);
          const auto grad_folded = grad.reshape({batch_size, n / 2, 2, 2});
          d_input = d_input.view({batch_size, n/ 2, 2, 2});
          // Accessors
          const auto input_a = input_folded.packed_accessor<scalar_t, 4>();
          const auto permuted_input_a = permuted_input.packed_accessor<scalar_t, 4>();
          const auto grad_reshaped_a = grad_reshaped.packed_accessor<scalar_t, 4>();
          const auto grad_a = grad_folded.packed_accessor<scalar_t, 4>();
          const auto permuted_grad_a = permuted_grad.packed_accessor<scalar_t, 4>();
          auto d_input_a = d_input.packed_accessor<scalar_t, 4>();
          permutation_factor_even_odd_multiply_complex_backward_cuda_kernel<scalar_t>
            <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(grad_a, grad_reshaped_a, permuted_grad_a, p_a, input_a, permuted_input_a, d_p_expanded_a, d_input_a);
          d_input = d_input.view({batch_size, n, 2});
          break;
        }
      default:
        AT_ERROR("permutation_factor_even_odd_multiply_backward requires input dimension 2 or 3");
    }
  });
  AT_CHECK(cudaGetLastError() == cudaSuccess,
     "permutation_factor_even_odd_multiply_backward_cuda failed with error code ",
     cudaGetLastError());
}

template <typename scalar_t>
__global__ void permutation_factor_reverse_multiply_cuda_kernel(const at::PackedTensorAccessor<scalar_t, 1> p_a,
                                                                const at::PackedTensorAccessor<scalar_t, 3> input_a,
                                                                at::PackedTensorAccessor<scalar_t, 3> output_a) {
  const scalar_t p[2] = {p_a[0], p_a[1]};
  const auto batch_size = input_a.size(0);
  const auto n = input_a.size(2);  // already divided by 2
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n / 2; i += blockDim.x * gridDim.x) {
    for (int64_t b = blockIdx.y * blockDim.y + threadIdx.y; b < batch_size; b += blockDim.y * gridDim.y) {
      #pragma unroll
      for (int j = 0; j <= 1; ++j) {
        const scalar_t in[2] = {input_a[b][j][i], input_a[b][j][n - 1 - i]};
        output_a[b][j][i] = (1 - p[j]) * in[0] + p[j] * in[1];
        output_a[b][j][n - 1 - i] = p[j] * in[0] + (1 - p[j]) * in[1];
      }
    }
  }
}

template <typename scalar_t>
__global__ void permutation_factor_reverse_multiply_complex_cuda_kernel(const at::PackedTensorAccessor<scalar_t, 1> p_a,
                                                                        const at::PackedTensorAccessor<scalar_t, 4> input_a,
                                                                        at::PackedTensorAccessor<scalar_t, 4> output_a) {
  const scalar_t p[2] = {p_a[0], p_a[1]};
  const auto batch_size = input_a.size(0);
  const auto n = input_a.size(2);
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n / 2; i += blockDim.x * gridDim.x) {
    for (int64_t b = blockIdx.y * blockDim.y + threadIdx.y; b < batch_size; b += blockDim.y * gridDim.y) {
      #pragma unroll
      for (int j = 0; j <= 1; ++j) {
        #pragma unroll
        for (int k = 0; k <= 1; ++k) {
          const scalar_t in[2] = {input_a[b][j][i][k], input_a[b][j][n - 1 - i][k]};
          output_a[b][j][i][k] = (1 - p[j]) * in[0] + p[j] * in[1];
          output_a[b][j][n - 1 - i][k] = p[j] * in[0] + (1 - p[j]) * in[1];
        }
      }
    }
  }
}

void permutation_factor_reverse_multiply_cuda(const at::Tensor& p, const at::Tensor& input, at::Tensor& output) {
  const auto batch_size = input.size(0);
  const auto n = input.size(1);
  dim3 block;
  block.x = std::min<int64_t>(MAX_BLOCK_SIZE, n / 2);
  block.y = div_up(MAX_BLOCK_SIZE, block.x);
  dim3 grid(div_up(n / 4, block.x), div_up(batch_size, block.y * WORK_PER_THREAD));
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "permutation_factor_reverse_multiply", [&] {
    const auto p_a = p.packed_accessor<scalar_t, 1>();
    switch (input.dim()) {
      case 2: // real
        {
          const auto input_folded = input.reshape({batch_size, 2, n / 2});
          output = output.view({batch_size, 2, n / 2});
          const auto input_a = input_folded.packed_accessor<scalar_t, 3>();
          auto output_a = output.packed_accessor<scalar_t, 3>();
          permutation_factor_reverse_multiply_cuda_kernel<scalar_t>
            <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(p_a, input_a, output_a);
          output = output.view({batch_size, n});
          break;
        }
      case 3: // complex
        {
          const auto input_folded = input.reshape({batch_size, 2, n / 2, 2});
          output = output.view({batch_size, 2, n / 2, 2});
          const auto input_a = input_folded.packed_accessor<scalar_t, 4>();
          auto output_a = output.packed_accessor<scalar_t, 4>();
          permutation_factor_reverse_multiply_complex_cuda_kernel<scalar_t>
            <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(p_a, input_a, output_a);
          output = output.view({batch_size, n, 2});
          break;
        }
      default:
        AT_ERROR("permutation_factor_reverse_multiply requires input dimension 2 or 3");
    }
  });
  AT_CHECK(cudaGetLastError() == cudaSuccess,
     "permutation_factor_reverse_multiply_cuda failed with error code ",
     cudaGetLastError());
}

template <typename scalar_t>
__global__ void permutation_factor_reverse_multiply_backward_cuda_kernel(const at::PackedTensorAccessor<scalar_t, 3> grad_a,
                                                                         const at::PackedTensorAccessor<scalar_t, 1> p_a,
                                                                         const at::PackedTensorAccessor<scalar_t, 3> input_a,
                                                                         at::PackedTensorAccessor<scalar_t, 3> d_p_expanded_a,
                                                                         at::PackedTensorAccessor<scalar_t, 3> d_input_a) {
  const scalar_t p[2] = {p_a[0], p_a[1]};
  const auto batch_size = input_a.size(0);
  const auto n = input_a.size(2);
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n / 2; i += blockDim.x * gridDim.x) {
    for (int64_t b = blockIdx.y * blockDim.y + threadIdx.y; b < batch_size; b += blockDim.y * gridDim.y) {
      #pragma unroll
      for (int j = 0; j <= 1; ++j) {
        const scalar_t in[2] = {input_a[b][j][i], input_a[b][j][n - 1 - i]};
        const scalar_t g[2] = {grad_a[b][j][i], grad_a[b][j][n - 1 - i]};
        d_p_expanded_a[j][b][i] = (in[1] - in[0]) * (g[0] - g[1]);
        d_input_a[b][j][i] = (1 - p[j]) * g[0] + p[j] * g[1];
        d_input_a[b][j][n - 1 - i] = p[j] * g[0] + (1 - p[j]) * g[1];
      }
    }
  }
}

template <typename scalar_t>
__global__ void permutation_factor_reverse_multiply_complex_backward_cuda_kernel(const at::PackedTensorAccessor<scalar_t, 4> grad_a,
                                                                                 const at::PackedTensorAccessor<scalar_t, 1> p_a,
                                                                                 const at::PackedTensorAccessor<scalar_t, 4> input_a,
                                                                                 at::PackedTensorAccessor<scalar_t, 3> d_p_expanded_a,
                                                                                 at::PackedTensorAccessor<scalar_t, 4> d_input_a) {
  const scalar_t p[2] = {p_a[0], p_a[1]};
  const auto batch_size = input_a.size(0);
  const auto n = input_a.size(2);
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n / 2; i += blockDim.x * gridDim.x) {
    for (int64_t b = blockIdx.y * blockDim.y + threadIdx.y; b < batch_size; b += blockDim.y * gridDim.y) {
      #pragma unroll
      for (int j = 0; j <= 1; ++j) {
        scalar_t d_p_expanded_temp = 0;
        #pragma unroll
        for (int k = 0; k <= 1; ++k) {
          const scalar_t in[2] = {input_a[b][j][i][k], input_a[b][j][n - 1 - i][k]};
          const scalar_t g[2] = {grad_a[b][j][i][k], grad_a[b][j][n - 1 - i][k]};
          d_p_expanded_temp += (in[1] - in[0]) * (g[0] - g[1]);
          d_input_a[b][j][i][k] = (1 - p[j]) * g[0] + p[j] * g[1];
          d_input_a[b][j][n - 1 - i][k] = p[j] * g[0] + (1 - p[j]) * g[1];
        }
        d_p_expanded_a[j][b][i] = d_p_expanded_temp;
      }
    }
  }
}

void permutation_factor_reverse_multiply_backward_cuda(const at::Tensor& grad, const at::Tensor& p, const at::Tensor& input,
                                                       at::Tensor& d_p_expanded, at::Tensor& d_input) {
  const auto batch_size = input.size(0);
  const auto n = input.size(1);
  dim3 block;
  block.x = std::min<int64_t>(MAX_BLOCK_SIZE, n / 2);
  block.y = div_up(MAX_BLOCK_SIZE, block.x);
  dim3 grid(div_up(n / 4, block.x), div_up(batch_size, block.y * WORK_PER_THREAD));
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "permutation_factor_reverse_multiply_backward", [&] {
    const auto p_a = p.packed_accessor<scalar_t, 1>();
    auto d_p_expanded_a = d_p_expanded.packed_accessor<scalar_t, 3>();
    switch (input.dim()) {
      case 2: // real
        {
          const auto input_folded = input.reshape({batch_size, 2, n / 2});
          const auto grad_folded = grad.reshape({batch_size, 2, n / 2});
          d_input = d_input.view({batch_size, 2, n/ 2});
          // Accessors
          const auto input_a = input_folded.packed_accessor<scalar_t, 3>();
          const auto grad_a = grad_folded.packed_accessor<scalar_t, 3>();
          auto d_input_a = d_input.packed_accessor<scalar_t, 3>();
          permutation_factor_reverse_multiply_backward_cuda_kernel<scalar_t>
            <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(grad_a, p_a, input_a, d_p_expanded_a, d_input_a);
          d_input = d_input.view({batch_size, n});
          break;
        }
      case 3: // complex
        {
          const auto input_folded = input.reshape({batch_size, 2, n / 2, 2});
          const auto grad_folded = grad.reshape({batch_size, 2, n / 2, 2});
          d_input = d_input.view({batch_size, 2, n/ 2, 2});
          // Accessors
          const auto input_a = input_folded.packed_accessor<scalar_t, 4>();
          const auto grad_a = grad_folded.packed_accessor<scalar_t, 4>();
          auto d_input_a = d_input.packed_accessor<scalar_t, 4>();
          permutation_factor_reverse_multiply_complex_backward_cuda_kernel<scalar_t>
            <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(grad_a, p_a, input_a, d_p_expanded_a, d_input_a);
          d_input = d_input.view({batch_size, n, 2});
          break;
        }
      default:
        AT_ERROR("permutation_factor_reverse_multiply_backward requires input dimension 2 or 3");
    }
  });
  AT_CHECK(cudaGetLastError() == cudaSuccess,
     "permutation_factor_reverse_multiply_backward_cuda failed with error code ",
     cudaGetLastError());
}
