#include <stdio.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THCAtomics.cuh>  // For atomicAdd on Half

#define FULL_MASK 0xffffffff

static constexpr int MAX_BLOCK_SIZE = 1024;
static constexpr int WORK_PER_THREAD = 16;
static constexpr int ELEMENTARY_SIZE = MAX_BLOCK_SIZE / 2;

__host__ __device__ static inline int64_t div_up(int64_t a, int64_t b) {
  return (a + b - 1) / b;
}

__host__ __device__ static inline int div_up(int a, int b) {
  return (a + b - 1) / b;
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
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "butterfly_factor_multiply_cuda", [&] {
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
  // AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "butterfly_factor_multiply_backward_cuda", [&] {
  AT_DISPATCH_FLOATING_TYPES(input.type(), "butterfly_factor_multiply_backward_cuda", [&] {
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
__global__ void butterfly_factor_multiply_inplace_cuda_kernel(const at::PackedTensorAccessor<scalar_t, 3> twiddle_a,
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
      int low_order_bits = i % stride;
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
__global__ void butterfly_factor_multiply_inplace_onestep_cuda_kernel(const at::PackedTensorAccessor<scalar_t, 3> twiddle_a,
                                                                      at::PackedTensorAccessor<scalar_t, 2> input_a,
                                                                      int stride) {
  const int batch_size = input_a.size(0);
  int twiddle_start_idx = stride - 1;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int low_order_bits = i % stride;
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

void butterfly_factor_multiply_inplace_cuda(const at::Tensor& twiddle, at::Tensor& input) {
  const int batch_size = input.size(0);
  const int n = input.size(1);
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "butterfly_factor_multiply_inplace_cuda", [&] {
    switch (input.dim()) {
      case 2:  // real
        {
          const auto twiddle_a = twiddle.packed_accessor<scalar_t, 3>();
          auto input_a = input.packed_accessor<scalar_t, 2>();
          int stride = std::min<int>(ELEMENTARY_SIZE, n / 2);
          dim3 block(stride);
          dim3 grid(div_up(n / 2, stride), batch_size);
          butterfly_factor_multiply_inplace_cuda_kernel<scalar_t>
            <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, input_a, stride);
          for (stride *= 2; stride <= n / 2; stride *= 2) {
            dim3 block(MAX_BLOCK_SIZE / 2);
            dim3 grid(div_up(n / 2, MAX_BLOCK_SIZE / 2), div_up(batch_size, WORK_PER_THREAD));
            butterfly_factor_multiply_inplace_onestep_cuda_kernel<scalar_t>
              <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, input_a, stride);
          }
          break;
        }
      case 3:  // complex
        {
          const auto twiddle_a = twiddle.packed_accessor<scalar_t, 4>();
          auto input_a = input.packed_accessor<scalar_t, 3>();
          AT_ERROR("Not implemented");
          // butterfly_factor_multiply_inplace_complex_cuda_kernel<scalar_t>
          //   <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, input_a, output_a);
          break;
        }
      default:
        AT_ERROR("butterfly_factor_multiply_inplace requires input dimension 2 or 3");
    }
  });
  AT_CHECK(cudaGetLastError() == cudaSuccess,
     "butterfly_factor_multiply_inplace_cuda failed with error code ",
     cudaGetLastError());
}

template <typename scalar_t>
__global__ void butterfly_factor_multiply_inplace_backward_cuda_kernel(const at::PackedTensorAccessor<scalar_t, 3> twiddle_a,
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
      int low_order_bits = i % stride;
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
      // Warp reduction
      for (int offset = warpSize / 2; offset >= stride; offset /= 2) {
        d_twiddle_val[0][0] += __shfl_down_sync(FULL_MASK, d_twiddle_val[0][0], offset);
        d_twiddle_val[0][1] += __shfl_down_sync(FULL_MASK, d_twiddle_val[0][1], offset);
        d_twiddle_val[1][0] += __shfl_down_sync(FULL_MASK, d_twiddle_val[1][0], offset);
        d_twiddle_val[1][1] += __shfl_down_sync(FULL_MASK, d_twiddle_val[1][1], offset);
      }
      // Block reduction
      int tid = threadIdx.x + threadIdx.y * blockDim.x;
      int nthreads = blockDim.x * blockDim.y;
      int lane = tid % warpSize;
      // if (lane < stride) {
      //   atomicAdd(&d_twiddle_a[twiddle_start_idx + twiddle_idx][0][0], d_twiddle_val[0][0]);
      //   atomicAdd(&d_twiddle_a[twiddle_start_idx + twiddle_idx][0][1], d_twiddle_val[0][1]);
      //   atomicAdd(&d_twiddle_a[twiddle_start_idx + twiddle_idx][1][0], d_twiddle_val[1][0]);
      //   atomicAdd(&d_twiddle_a[twiddle_start_idx + twiddle_idx][1][1], d_twiddle_val[1][1]);
      // }
      if (stride < nthreads) {
        __syncthreads();
        s_d_twiddle[tid] = 0;
        s_d_twiddle[tid + ELEMENTARY_SIZE] = 0;
        s_d_twiddle[tid + 2 * ELEMENTARY_SIZE] = 0;
        s_d_twiddle[tid + 3 * ELEMENTARY_SIZE] = 0;
        __syncthreads();
        if (lane < stride) {
          atomicAdd(&s_d_twiddle[twiddle_idx], d_twiddle_val[0][0]);
          atomicAdd(&s_d_twiddle[twiddle_idx + ELEMENTARY_SIZE], d_twiddle_val[0][1]);
          atomicAdd(&s_d_twiddle[twiddle_idx + 2 * ELEMENTARY_SIZE], d_twiddle_val[1][0]);
          atomicAdd(&s_d_twiddle[twiddle_idx + 3 * ELEMENTARY_SIZE], d_twiddle_val[1][1]);
        }
        __syncthreads();
        if (tid < stride) {
          atomicAdd(&d_twiddle_a[twiddle_start_idx + twiddle_idx][0][0], s_d_twiddle[twiddle_idx]);
          atomicAdd(&d_twiddle_a[twiddle_start_idx + twiddle_idx][0][1], s_d_twiddle[twiddle_idx + ELEMENTARY_SIZE]);
          atomicAdd(&d_twiddle_a[twiddle_start_idx + twiddle_idx][1][0], s_d_twiddle[twiddle_idx + 2 * ELEMENTARY_SIZE]);
          atomicAdd(&d_twiddle_a[twiddle_start_idx + twiddle_idx][1][1], s_d_twiddle[twiddle_idx + 3 * ELEMENTARY_SIZE]);
        }
      } else {
        atomicAdd(&d_twiddle_a[twiddle_start_idx + twiddle_idx][0][0], d_twiddle_val[0][0]);
        atomicAdd(&d_twiddle_a[twiddle_start_idx + twiddle_idx][0][1], d_twiddle_val[0][1]);
        atomicAdd(&d_twiddle_a[twiddle_start_idx + twiddle_idx][1][0], d_twiddle_val[1][0]);
        atomicAdd(&d_twiddle_a[twiddle_start_idx + twiddle_idx][1][1], d_twiddle_val[1][1]);
      }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < max_stride * 2; i += blockDim.x) {
      d_input_a[b][input_base_idx + i] = s_grad[i];
    }
  }
}

template <typename scalar_t>
__global__ void butterfly_factor_multiply_inplace_backward_onestep_cuda_kernel(const at::PackedTensorAccessor<scalar_t, 3> twiddle_a,
                                                                               at::PackedTensorAccessor<double, 2> output_a,
                                                                               at::PackedTensorAccessor<scalar_t, 3> d_twiddle_a,
                                                                               at::PackedTensorAccessor<scalar_t, 2> d_input_a,
                                                                               int stride) {
  const int batch_size = output_a.size(0);
  const int n = output_a.size(1);
  int twiddle_start_idx = stride - 1;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i > n) return;
  int low_order_bits = i % stride;
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

void butterfly_factor_multiply_inplace_backward_cuda(const at::Tensor& grad, const at::Tensor& twiddle, at::Tensor& output,
                                                     at::Tensor& d_twiddle, at::Tensor& d_input) {
  const int batch_size = output.size(0);
  const int n = output.size(1);
  AT_DISPATCH_FLOATING_TYPES(grad.type(), "butterfly_factor_multiply_inplace_backward_cuda", [&] {
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
            butterfly_factor_multiply_inplace_backward_onestep_cuda_kernel<scalar_t>
              <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, output_a, d_twiddle_a, d_input_a, stride);
          }
          dim3 block(stride);
          dim3 grid(div_up(n / 2, stride), batch_size);
          butterfly_factor_multiply_inplace_backward_cuda_kernel<scalar_t>
            <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, output_a, d_twiddle_a, d_input_a, stride);
          break;
        }
      case 3:  // complex
        {
          const auto twiddle_a = twiddle.packed_accessor<scalar_t, 4>();
          auto output_a = output.packed_accessor<scalar_t, 3>();
          AT_ERROR("Not implemented");
          // butterfly_factor_multiply_inplace_backward_complex_cuda_kernel<scalar_t>
          //   <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, output_a, output_a);
          break;
        }
      default:
        AT_ERROR("butterfly_factor_multiply_inplace_backward requires input dimension 2 or 3");
    }
  });
  AT_CHECK(cudaGetLastError() == cudaSuccess,
     "butterfly_factor_multiply_inplace_backward_cuda failed with error code ",
     cudaGetLastError());
}

template <typename scalar_t>
__global__ void butterfly_factor_multiply_intermediate_cuda_kernel(const at::PackedTensorAccessor<scalar_t, 3> twiddle_a,
                                                                   at::PackedTensorAccessor<scalar_t, 3> output_a,
                                                                   int max_log_stride) {
  const int batch_size = output_a.size(1);
  const int max_stride = 1 << max_log_stride;
  const int input_base_idx = blockIdx.x * blockDim.x * 2;
  __shared__ scalar_t s_input[ELEMENTARY_SIZE * 2];
  __shared__ scalar_t s_twiddle[ELEMENTARY_SIZE][2][2];
  int64_t b = blockIdx.y * blockDim.y + threadIdx.y;
  if (b < batch_size) {  // Currently we assume 1 batch per thread block, so all threads in the block should enter (otherwise deadlock)
    for (int i = threadIdx.x; i < max_stride * 2; i += blockDim.x) {
      s_input[i] = output_a[0][b][input_base_idx + i];
    }
    int i = threadIdx.x;
    for (int log_stride = 0; log_stride <= max_log_stride; ++log_stride) {
      int stride = 1 << log_stride;
      int twiddle_start_idx = stride - 1;
      if (i < stride) {
        s_twiddle[i][0][0] = twiddle_a[twiddle_start_idx + i][0][0];
        s_twiddle[i][0][1] = twiddle_a[twiddle_start_idx + i][0][1];
        s_twiddle[i][1][0] = twiddle_a[twiddle_start_idx + i][1][0];
        s_twiddle[i][1][1] = twiddle_a[twiddle_start_idx + i][1][1];
      }
      int low_order_bits = i % stride;
      int twiddle_idx = low_order_bits;
      int pos = 2 * (i - low_order_bits) + low_order_bits;
      __syncthreads();
      const scalar_t twiddle_val[2][2] = {{s_twiddle[twiddle_idx][0][0], s_twiddle[twiddle_idx][0][1]},
                                          {s_twiddle[twiddle_idx][1][0], s_twiddle[twiddle_idx][1][1]}};
      const scalar_t input_val[2] = {s_input[pos], s_input[pos + stride]};
      s_input[pos] = twiddle_val[0][0] * input_val[0] + twiddle_val[0][1] * input_val[1];
      s_input[pos + stride] = twiddle_val[1][0] * input_val[0] + twiddle_val[1][1] * input_val[1];
      output_a[log_stride+1][b][input_base_idx + pos] = s_input[pos];
      output_a[log_stride+1][b][input_base_idx + pos + stride] = s_input[pos + stride];
    }
  }
}

template <typename scalar_t>
__global__ void butterfly_factor_multiply_intermediate_complex_cuda_kernel(const at::PackedTensorAccessor<scalar_t, 4> twiddle_a,
                                                                           at::PackedTensorAccessor<scalar_t, 4> output_a,
                                                                           int max_log_stride) {
  const int batch_size = output_a.size(1);
  const int max_stride = 1 << max_log_stride;
  const int input_base_idx = blockIdx.x * blockDim.x * 2;
  __shared__ scalar_t s_input[ELEMENTARY_SIZE * 2][2];
  __shared__ scalar_t s_twiddle[ELEMENTARY_SIZE][2][2][2];
  int64_t b = blockIdx.y * blockDim.y + threadIdx.y;
  if (b < batch_size) {  // Currently we assume 1 batch per thread block, so all threads in the block should enter (otherwise deadlock)
    for (int i = threadIdx.x; i < max_stride * 2; i += blockDim.x) {
      s_input[i][0] = output_a[0][b][input_base_idx + i][0];
      s_input[i][1] = output_a[0][b][input_base_idx + i][1];
    }
    int i = threadIdx.x;
    for (int log_stride = 0; log_stride <= max_log_stride; ++log_stride) {
      int stride = 1 << log_stride;
      int twiddle_start_idx = stride - 1;
      if (i < stride) {
        s_twiddle[i][0][0][0] = twiddle_a[twiddle_start_idx + i][0][0][0];
        s_twiddle[i][0][0][1] = twiddle_a[twiddle_start_idx + i][0][0][1];
        s_twiddle[i][0][1][0] = twiddle_a[twiddle_start_idx + i][0][1][0];
        s_twiddle[i][0][1][1] = twiddle_a[twiddle_start_idx + i][0][1][1];
        s_twiddle[i][1][0][0] = twiddle_a[twiddle_start_idx + i][1][0][0];
        s_twiddle[i][1][0][1] = twiddle_a[twiddle_start_idx + i][1][0][1];
        s_twiddle[i][1][1][0] = twiddle_a[twiddle_start_idx + i][1][1][0];
        s_twiddle[i][1][1][1] = twiddle_a[twiddle_start_idx + i][1][1][1];
      }
      int low_order_bits = i % stride;
      int twiddle_idx = low_order_bits;
      int pos = 2 * (i - low_order_bits) + low_order_bits;
      __syncthreads();
      const scalar_t twiddle_val[2][2][2] = {{{s_twiddle[twiddle_idx][0][0][0], s_twiddle[twiddle_idx][0][0][1]},
                                              {s_twiddle[twiddle_idx][0][1][0], s_twiddle[twiddle_idx][0][1][1]}},
                                             {{s_twiddle[twiddle_idx][1][0][0], s_twiddle[twiddle_idx][1][0][1]},
                                              {s_twiddle[twiddle_idx][1][1][0], s_twiddle[twiddle_idx][1][1][1]}}};
      const scalar_t input_val[2][2] = {{s_input[pos][0], s_input[pos][1]},
                                        {s_input[pos + stride][0], s_input[pos + stride][1]}};
      s_input[pos][0] = twiddle_val[0][0][0] * input_val[0][0] - twiddle_val[0][0][1] * input_val[0][1]
        + twiddle_val[0][1][0] * input_val[1][0] - twiddle_val[0][1][1] * input_val[1][1];
      s_input[pos][1] = twiddle_val[0][0][0] * input_val[0][1] + twiddle_val[0][0][1] * input_val[0][0]
        + twiddle_val[0][1][0] * input_val[1][1] + twiddle_val[0][1][1] * input_val[1][0];
      s_input[pos + stride][0] = twiddle_val[1][0][0] * input_val[0][0] - twiddle_val[1][0][1] * input_val[0][1]
        + twiddle_val[1][1][0] * input_val[1][0] - twiddle_val[1][1][1] * input_val[1][1];
      s_input[pos + stride][1] = twiddle_val[1][0][0] * input_val[0][1] + twiddle_val[1][0][1] * input_val[0][0]
        + twiddle_val[1][1][0] * input_val[1][1] + twiddle_val[1][1][1] * input_val[1][0];
      output_a[log_stride+1][b][input_base_idx + pos][0] = s_input[pos][0];
      output_a[log_stride+1][b][input_base_idx + pos][1] = s_input[pos][1];
      output_a[log_stride+1][b][input_base_idx + pos + stride][0] = s_input[pos + stride][0];
      output_a[log_stride+1][b][input_base_idx + pos + stride][1] = s_input[pos + stride][1];

    }
  }
}

template <typename scalar_t>
__global__ void butterfly_factor_multiply_intermediate_onestep_cuda_kernel(const at::PackedTensorAccessor<scalar_t, 3> twiddle_a,
                                                                           at::PackedTensorAccessor<scalar_t, 3> output_a,
                                                                           int log_stride) {
  const int batch_size = output_a.size(1);
  const int stride = 1 << log_stride;
  int twiddle_start_idx = stride - 1;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int low_order_bits = i % stride;
  int twiddle_idx = twiddle_start_idx + low_order_bits;
  int pos = 2 * (i - low_order_bits) + low_order_bits;
  const scalar_t twiddle_val[2][2] = {{twiddle_a[twiddle_idx][0][0], twiddle_a[twiddle_idx][0][1]},
                                      {twiddle_a[twiddle_idx][1][0], twiddle_a[twiddle_idx][1][1]}};
  for (int64_t b = blockIdx.y * blockDim.y + threadIdx.y; b < batch_size; b += blockDim.y * gridDim.y) {
    const scalar_t input_val[2] = {output_a[log_stride][b][pos], output_a[log_stride][b][pos + stride]};
    output_a[log_stride+1][b][pos] = twiddle_val[0][0] * input_val[0] + twiddle_val[0][1] * input_val[1];
    output_a[log_stride+1][b][pos + stride] = twiddle_val[1][0] * input_val[0] + twiddle_val[1][1] * input_val[1];
  }
}

template <typename scalar_t>
__global__ void butterfly_factor_multiply_intermediate_onestep_complex_cuda_kernel(const at::PackedTensorAccessor<scalar_t, 4> twiddle_a,
                                                                                   at::PackedTensorAccessor<scalar_t, 4> output_a,
                                                                                   int log_stride) {
  const int batch_size = output_a.size(1);
  const int stride = 1 << log_stride;
  int twiddle_start_idx = stride - 1;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int low_order_bits = i % stride;
  int twiddle_idx = twiddle_start_idx + low_order_bits;
  int pos = 2 * (i - low_order_bits) + low_order_bits;
  const scalar_t twiddle_val[2][2][2] = {{{twiddle_a[twiddle_idx][0][0][0], twiddle_a[twiddle_idx][0][0][1]},
                                          {twiddle_a[twiddle_idx][0][1][0], twiddle_a[twiddle_idx][0][1][1]}},
                                         {{twiddle_a[twiddle_idx][1][0][0], twiddle_a[twiddle_idx][1][0][1]},
                                          {twiddle_a[twiddle_idx][1][1][0], twiddle_a[twiddle_idx][1][1][1]}}};
  for (int64_t b = blockIdx.y * blockDim.y + threadIdx.y; b < batch_size; b += blockDim.y * gridDim.y) {
    const scalar_t input_val[2][2] = {{output_a[log_stride][b][pos][0], output_a[log_stride][b][pos][1]},
                                      {output_a[log_stride][b][pos + stride][0], output_a[log_stride][b][pos + stride][1]}};
    output_a[log_stride+1][b][pos][0] = twiddle_val[0][0][0] * input_val[0][0] - twiddle_val[0][0][1] * input_val[0][1]
      + twiddle_val[0][1][0] * input_val[1][0] - twiddle_val[0][1][1] * input_val[1][1];
    output_a[log_stride+1][b][pos][1] = twiddle_val[0][0][0] * input_val[0][1] + twiddle_val[0][0][1] * input_val[0][0]
      + twiddle_val[0][1][0] * input_val[1][1] + twiddle_val[0][1][1] * input_val[1][0];
    output_a[log_stride+1][b][pos + stride][0] = twiddle_val[1][0][0] * input_val[0][0] - twiddle_val[1][0][1] * input_val[0][1]
      + twiddle_val[1][1][0] * input_val[1][0] - twiddle_val[1][1][1] * input_val[1][1];
    output_a[log_stride+1][b][pos + stride][1] = twiddle_val[1][0][0] * input_val[0][1] + twiddle_val[1][0][1] * input_val[0][0]
      + twiddle_val[1][1][0] * input_val[1][1] + twiddle_val[1][1][1] * input_val[1][0];
  }
}

void butterfly_factor_multiply_intermediate_cuda(const at::Tensor& twiddle, at::Tensor& output) {
  const int batch_size = output.size(1);
  const int n = output.size(2);
  const int log_n = int(log2((double) n));
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(output.type(), "butterfly_factor_multiply_intermediate_cuda", [&] {
    switch (output.dim()) {
      case 3:  // real
        {
          const auto twiddle_a = twiddle.packed_accessor<scalar_t, 3>();
          auto output_a = output.packed_accessor<scalar_t, 3>();
          int stride = std::min<int>(ELEMENTARY_SIZE, n / 2);
          int log_stride = int(log2((double) stride));
          dim3 block(stride);
          dim3 grid(div_up(n / 2, stride), batch_size);
          butterfly_factor_multiply_intermediate_cuda_kernel<scalar_t>
            <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, output_a, log_stride);
          // log_stride = -1;
          for (log_stride++; log_stride <= log_n - 1; ++log_stride) {
            stride = 1 << log_stride;
            dim3 block(MAX_BLOCK_SIZE / 2);
            dim3 grid(div_up(n / 2, MAX_BLOCK_SIZE / 2), div_up(batch_size, WORK_PER_THREAD));
            butterfly_factor_multiply_intermediate_onestep_cuda_kernel<scalar_t>
              <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, output_a, log_stride);
          }
          break;
        }
      case 4:  // complex
        {
          const auto twiddle_a = twiddle.packed_accessor<scalar_t, 4>();
          auto output_a = output.packed_accessor<scalar_t, 4>();
          int stride = std::min<int>(ELEMENTARY_SIZE, n / 2);
          int log_stride = int(log2((double) stride));
          dim3 block(stride);
          dim3 grid(div_up(n / 2, stride), batch_size);
          butterfly_factor_multiply_intermediate_complex_cuda_kernel<scalar_t>
            <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, output_a, log_stride);
          for (log_stride++; log_stride <= log_n - 1; ++log_stride) {
            stride = 1 << log_stride;
            dim3 block(MAX_BLOCK_SIZE / 2);
            dim3 grid(div_up(n / 2, MAX_BLOCK_SIZE / 2), div_up(batch_size, WORK_PER_THREAD));
            butterfly_factor_multiply_intermediate_onestep_complex_cuda_kernel<scalar_t>
              <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, output_a, log_stride);
          }
          break;
        }
      default:
        AT_ERROR("butterfly_factor_multiply_intermediate requires input dimension 2 or 3");
    }
  });
  AT_CHECK(cudaGetLastError() == cudaSuccess,
     "butterfly_factor_multiply_intermediate_cuda failed with error code ",
     cudaGetLastError());
}

template <typename scalar_t>
__global__ void butterfly_factor_multiply_intermediate_backward_cuda_kernel(const at::PackedTensorAccessor<scalar_t, 3> twiddle_a,
                                                                            const at::PackedTensorAccessor<scalar_t, 3> output_a,
                                                                            at::PackedTensorAccessor<scalar_t, 3> d_twiddle_a,
                                                                            at::PackedTensorAccessor<scalar_t, 2> d_input_a,
                                                                            int log_max_stride) {
  const int batch_size = output_a.size(1);
  const int max_stride = 1 << log_max_stride;
  const int input_base_idx = blockIdx.x * blockDim.x * 2;
  __shared__ scalar_t s_grad[ELEMENTARY_SIZE * 2];
  __shared__ scalar_t s_twiddle[ELEMENTARY_SIZE][2][2];
  // __shared__ scalar_t s_d_twiddle[ELEMENTARY_SIZE * 4];
  scalar_t* s_d_twiddle = &s_twiddle[0][0][0];  // Reusing the same storage as s_twiddle, have to be careful if we change the implemetnation.
  int64_t b = blockIdx.y * blockDim.y + threadIdx.y;
  if (b < batch_size) {  // Currently we assume 1 batch per thread block, so all threads in the block should enter (otherwise deadlock)
    for (int i = threadIdx.x; i < max_stride * 2; i += blockDim.x) {
      s_grad[i] = d_input_a[b][input_base_idx + i];
    }
    int i = threadIdx.x;
    for (int log_stride = log_max_stride; log_stride >= 0; --log_stride) {
      int stride = 1 << log_stride;
      int twiddle_start_idx = stride - 1;
      if (i < stride) {
        s_twiddle[i][0][0] = twiddle_a[twiddle_start_idx + i][0][0];
        s_twiddle[i][0][1] = twiddle_a[twiddle_start_idx + i][0][1];
        s_twiddle[i][1][0] = twiddle_a[twiddle_start_idx + i][1][0];
        s_twiddle[i][1][1] = twiddle_a[twiddle_start_idx + i][1][1];
      }
      int low_order_bits = i % stride;
      int twiddle_idx = low_order_bits;
      int pos = 2 * (i - low_order_bits) + low_order_bits;
      __syncthreads();
      const scalar_t twiddle_val[2][2] = {{s_twiddle[twiddle_idx][0][0], s_twiddle[twiddle_idx][0][1]},
                                          {s_twiddle[twiddle_idx][1][0], s_twiddle[twiddle_idx][1][1]}};
      const scalar_t grad_val[2] = {s_grad[pos], s_grad[pos + stride]};
      s_grad[pos] = twiddle_val[0][0] * grad_val[0] + twiddle_val[1][0] * grad_val[1];
      s_grad[pos + stride] = twiddle_val[0][1] * grad_val[0] + twiddle_val[1][1] * grad_val[1];
      const scalar_t input_val[2] = {output_a[log_stride][b][input_base_idx + pos], output_a[log_stride][b][input_base_idx + pos + stride]};
      scalar_t d_twiddle_val[2][2] = {{grad_val[0] * input_val[0], grad_val[0] * input_val[1]},
                                      {grad_val[1] * input_val[0], grad_val[1] * input_val[1]}};
      // Warp reduction
      for (int offset = warpSize / 2; offset >= stride; offset /= 2) {
        d_twiddle_val[0][0] += __shfl_down_sync(FULL_MASK, d_twiddle_val[0][0], offset);
        d_twiddle_val[0][1] += __shfl_down_sync(FULL_MASK, d_twiddle_val[0][1], offset);
        d_twiddle_val[1][0] += __shfl_down_sync(FULL_MASK, d_twiddle_val[1][0], offset);
        d_twiddle_val[1][1] += __shfl_down_sync(FULL_MASK, d_twiddle_val[1][1], offset);
      }
      // Block reduction
      int tid = threadIdx.x + threadIdx.y * blockDim.x;
      int nthreads = blockDim.x * blockDim.y;
      int lane = tid % warpSize;
      // if (lane < stride) {
      //   atomicAdd(&d_twiddle_a[twiddle_start_idx + twiddle_idx][0][0], d_twiddle_val[0][0]);
      //   atomicAdd(&d_twiddle_a[twiddle_start_idx + twiddle_idx][0][1], d_twiddle_val[0][1]);
      //   atomicAdd(&d_twiddle_a[twiddle_start_idx + twiddle_idx][1][0], d_twiddle_val[1][0]);
      //   atomicAdd(&d_twiddle_a[twiddle_start_idx + twiddle_idx][1][1], d_twiddle_val[1][1]);
      // }
      if (stride < nthreads) {
        __syncthreads();
        s_d_twiddle[tid] = 0;
        s_d_twiddle[tid + ELEMENTARY_SIZE] = 0;
        s_d_twiddle[tid + 2 * ELEMENTARY_SIZE] = 0;
        s_d_twiddle[tid + 3 * ELEMENTARY_SIZE] = 0;
        __syncthreads();
        if (lane < stride) {
          atomicAdd(&s_d_twiddle[twiddle_idx], d_twiddle_val[0][0]);
          atomicAdd(&s_d_twiddle[twiddle_idx + ELEMENTARY_SIZE], d_twiddle_val[0][1]);
          atomicAdd(&s_d_twiddle[twiddle_idx + 2 * ELEMENTARY_SIZE], d_twiddle_val[1][0]);
          atomicAdd(&s_d_twiddle[twiddle_idx + 3 * ELEMENTARY_SIZE], d_twiddle_val[1][1]);
        }
        __syncthreads();
        if (tid < stride) {
          atomicAdd(&d_twiddle_a[twiddle_start_idx + twiddle_idx][0][0], s_d_twiddle[twiddle_idx]);
          atomicAdd(&d_twiddle_a[twiddle_start_idx + twiddle_idx][0][1], s_d_twiddle[twiddle_idx + ELEMENTARY_SIZE]);
          atomicAdd(&d_twiddle_a[twiddle_start_idx + twiddle_idx][1][0], s_d_twiddle[twiddle_idx + 2 * ELEMENTARY_SIZE]);
          atomicAdd(&d_twiddle_a[twiddle_start_idx + twiddle_idx][1][1], s_d_twiddle[twiddle_idx + 3 * ELEMENTARY_SIZE]);
        }
      } else {
        atomicAdd(&d_twiddle_a[twiddle_start_idx + twiddle_idx][0][0], d_twiddle_val[0][0]);
        atomicAdd(&d_twiddle_a[twiddle_start_idx + twiddle_idx][0][1], d_twiddle_val[0][1]);
        atomicAdd(&d_twiddle_a[twiddle_start_idx + twiddle_idx][1][0], d_twiddle_val[1][0]);
        atomicAdd(&d_twiddle_a[twiddle_start_idx + twiddle_idx][1][1], d_twiddle_val[1][1]);
      }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < max_stride * 2; i += blockDim.x) {
      d_input_a[b][input_base_idx + i] = s_grad[i];
    }
  }
}

template <typename scalar_t>
__global__ void butterfly_factor_multiply_intermediate_backward_onestep_cuda_kernel(const at::PackedTensorAccessor<scalar_t, 3> twiddle_a,
                                                                                    const at::PackedTensorAccessor<scalar_t, 3> output_a,
                                                                                    at::PackedTensorAccessor<scalar_t, 3> d_twiddle_a,
                                                                                    at::PackedTensorAccessor<scalar_t, 2> d_input_a,
                                                                                    int log_stride) {
  const int batch_size = output_a.size(1);
  const int n = output_a.size(2);
  int stride = 1 << log_stride;
  int twiddle_start_idx = stride - 1;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i > n) return;
  int low_order_bits = i % stride;
  int twiddle_idx = twiddle_start_idx + low_order_bits;
  int pos = 2 * (i - low_order_bits) + low_order_bits;
  const scalar_t twiddle_val[2][2] = {{twiddle_a[twiddle_idx][0][0], twiddle_a[twiddle_idx][0][1]},
                                      {twiddle_a[twiddle_idx][1][0], twiddle_a[twiddle_idx][1][1]}};
  scalar_t d_twiddle_val[2][2] = {{0, 0}, {0, 0}};
  for (int64_t b = blockIdx.y * blockDim.y + threadIdx.y; b < batch_size; b += blockDim.y * gridDim.y) {
    const scalar_t grad_val[2] = {d_input_a[b][pos], d_input_a[b][pos + stride]};
    d_input_a[b][pos] = twiddle_val[0][0] * grad_val[0] + twiddle_val[1][0] * grad_val[1];
    d_input_a[b][pos + stride] = twiddle_val[0][1] * grad_val[0] + twiddle_val[1][1] * grad_val[1];
    const scalar_t input_val[2] = {output_a[log_stride][b][pos], output_a[log_stride][b][pos + stride]};
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

void butterfly_factor_multiply_intermediate_backward_cuda(const at::Tensor& twiddle, const at::Tensor& output,
                                                          at::Tensor& d_twiddle, at::Tensor& d_input) {
  const int batch_size = output.size(1);
  const int n = output.size(2);
  const int log_n = int(log2((double) n));
  AT_DISPATCH_FLOATING_TYPES(output.type(), "butterfly_factor_multiply_intermediate_backward_cuda", [&] {
    switch (output.dim()) {
      case 3:  // real
        {
          const auto twiddle_a = twiddle.packed_accessor<scalar_t, 3>();
          const auto output_a = output.packed_accessor<scalar_t, 3>();
          auto d_twiddle_a = d_twiddle.packed_accessor<scalar_t, 3>();
          auto d_input_a = d_input.packed_accessor<scalar_t, 2>();
          int stride = n/2;
          int log_stride = log_n - 1;
          for (; stride > ELEMENTARY_SIZE; stride /= 2) {
          // for (; stride > 0; stride /= 2) {
            log_stride = int(log2((double) stride));
            dim3 block(MAX_BLOCK_SIZE / 2);
            dim3 grid(div_up(n / 2, MAX_BLOCK_SIZE / 2), div_up(batch_size, WORK_PER_THREAD));
            butterfly_factor_multiply_intermediate_backward_onestep_cuda_kernel<scalar_t>
              <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, output_a, d_twiddle_a, d_input_a, log_stride);
          }
          log_stride = int(log2((double) stride));
          dim3 block(stride);
          dim3 grid(div_up(n / 2, stride), batch_size);
          butterfly_factor_multiply_intermediate_backward_cuda_kernel<scalar_t>
            <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, output_a, d_twiddle_a, d_input_a, log_stride);
          break;
        }
      case 4:  // complex
        {
          AT_ERROR("Not implemented");
          // const auto twiddle_a = twiddle.packed_accessor<scalar_t, 4>();
          // const auto output_a = output.packed_accessor<scalar_t, 3>();
          // butterfly_factor_multiply_intermediate_backward_complex_cuda_kernel<scalar_t>
          //   <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, output_a, output_a);
          break;
        }
      default:
        AT_ERROR("butterfly_factor_multiply_intermediate_backward requires input dimension 2 or 3");
    }
  });
  AT_CHECK(cudaGetLastError() == cudaSuccess,
     "butterfly_factor_multiply_intermediate_backward_cuda failed with error code ",
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
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "permutation_factor_even_odd_multiply", [&] {
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
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "permutation_factor_even_odd_multiply_backward", [&] {
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
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "permutation_factor_reverse_multiply", [&] {
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
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "permutation_factor_reverse_multiply_backward", [&] {
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
