#include "butterfly_cpu.h"

torch::Tensor butterfly_multiply_fw_cpu(const torch::Tensor& twiddle,
                                        const torch::Tensor& input,
                                        bool increasing_stride,
                                        bool return_intermediates) {
  return input;
}
