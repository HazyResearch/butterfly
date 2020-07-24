#include "butterfly_cuda.h"

torch::Tensor butterfly_multiply_fw_cuda(const torch::Tensor& twiddle,
                                         const torch::Tensor& input,
                                         bool increasing_stride,
                                         bool return_intermediates) {
  return input;
}