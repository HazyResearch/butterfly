#include "butterfly_cuda.h"

torch::Tensor butterfly_multiply_fw_cuda(const torch::Tensor twiddle,
                                         const torch::Tensor input,
                                         bool increasing_stride) {
  return input;
}

std::tuple<torch::Tensor, torch::Tensor>
  butterfly_multiply_bw_cuda(const torch::Tensor twiddle,
                             const torch::Tensor input,
                             const torch::Tensor grad,
                             bool increasing_stride) {
  return std::make_tuple(input, grad);
}