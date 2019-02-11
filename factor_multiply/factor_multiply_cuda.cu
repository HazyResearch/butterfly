#include <stdio.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

constexpr int MAX_BLOCK_SIZE = 1024;

template <typename scalar_t>
__global__ void butterfly_factor_multiply_cuda_kernel(const at::PackedTensorAccessor<scalar_t, 3> twiddle_a,
                                                      const at::PackedTensorAccessor<scalar_t, 3> input_a,
                                                      at::PackedTensorAccessor<scalar_t, 3> output_a) {
  const auto batch_size = input_a.size(0);
  const auto n = input_a.size(2);
  for (int64_t b = blockIdx.x * blockDim.y + threadIdx.y; b < batch_size; b += blockDim.y * gridDim.x) {
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
      const scalar_t twiddle_val[2][2] = {{twiddle_a[0][0][i], twiddle_a[0][1][i]},
                                          {twiddle_a[1][0][i], twiddle_a[1][1][i]}};
      const scalar_t input_val[2] = {input_a[b][0][i], input_a[b][1][i]};
      #pragma unroll
      for (int64_t j = 0; j <= 1; ++j) {
        output_a[b][j][i] = twiddle_val[j][0] * input_val[0] + twiddle_val[j][1] * input_val[1];
        // output_a[b][j][i] = twiddle_a[j][0][i] * input_a[b][0][i] + twiddle_a[j][1][i] * input_a[b][1][i];
      }
    }
  }
}

void butterfly_factor_multiply_cuda(const at::Tensor& twiddle, const at::Tensor& input, at::Tensor& output) {
  const auto batch_size = input.size(0);
  const auto n = input.size(2);
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "butterfly_factor_multiply_cuda", [&] {
      const auto twiddle_a = twiddle.packed_accessor<scalar_t, 3>();
      const auto input_a = input.packed_accessor<scalar_t, 3>();
      auto output_a = output.packed_accessor<scalar_t, 3>();
      dim3 block;
      block.x = std::min<int>(MAX_BLOCK_SIZE, n);
      block.y = (MAX_BLOCK_SIZE + block.x - 1) / block.x;
      // TODO: use grid to paralellize the n dimension, rn we're only using threads.
      const int grid = (batch_size + block.y - 1) / block.y;
      butterfly_factor_multiply_cuda_kernel<scalar_t>
        <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(twiddle_a, input_a, output_a);
  });
  AT_CHECK(cudaGetLastError() == cudaSuccess,
     "butterfly_factor_multiply_factor failed with error code ",
     cudaGetLastError());
}