#pragma once

#include <torch/extension.h>

torch::Tensor butterfly_multiply_fw_cpu(const torch::Tensor& twiddle,
                                        const torch::Tensor& input,
                                        bool increasing_stride);
