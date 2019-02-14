#include <stdio.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

static constexpr int MAX_BLOCK_SIZE = 512;
static constexpr int WORK_PER_THREAD = 4;

static inline int64_t div_up(int64_t a, int64_t b) {
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

template <typename scalar_t>
__global__ void butterfly_factor_multiply_backward_cuda_kernel(const at::PackedTensorAccessor<scalar_t, 3> grad_a,
                                                               const at::PackedTensorAccessor<scalar_t, 3> twiddle_a,
                                                               const at::PackedTensorAccessor<scalar_t, 3> input_a,
                                                               at::PackedTensorAccessor<scalar_t, 4> d_twiddle_expanded_a,
                                                               at::PackedTensorAccessor<scalar_t, 3> d_input_a) {
  const auto batch_size = input_a.size(0);
  const auto n = input_a.size(2);
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    const scalar_t twiddle_val[2][2] = {{twiddle_a[0][0][i], twiddle_a[0][1][i]},
                                        {twiddle_a[1][0][i], twiddle_a[1][1][i]}};
    for (int64_t b = blockIdx.y * blockDim.y + threadIdx.y; b < batch_size; b += blockDim.y * gridDim.y) {
      const scalar_t input_val[2] = {input_a[b][0][i], input_a[b][1][i]};
      const scalar_t grad_val[2] = {grad_a[b][0][i], grad_a[b][1][i]};
      #pragma unroll
      for (int j = 0; j <= 1; ++j) {
        d_twiddle_expanded_a[b][j][0][i] = grad_val[j] * input_val[0];
        d_twiddle_expanded_a[b][j][1][i] = grad_val[j] * input_val[1];
        d_input_a[b][j][i] = twiddle_val[0][j] * grad_val[0] + twiddle_val[1][j] * grad_val[1];
      }
    }
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
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "butterfly_factor_multiply_backward_cuda", [&] {
    switch (input.dim()) {
      case 3:  // real
        {
          const auto grad_a = grad.packed_accessor<scalar_t, 3>();
          const auto twiddle_a = twiddle.packed_accessor<scalar_t, 3>();
          const auto input_a = input.packed_accessor<scalar_t, 3>();
          auto d_twiddle_expanded_a = d_twiddle_expanded.packed_accessor<scalar_t, 4>();
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