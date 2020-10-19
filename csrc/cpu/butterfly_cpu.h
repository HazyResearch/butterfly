#pragma once

#include <torch/extension.h>
#include <tuple>

torch::Tensor butterfly_multiply_fw_cpu(const torch::Tensor twiddle,
                                        const torch::Tensor input,
                                        bool increasing_stride,
                                        int output_size);

std::tuple<torch::Tensor, torch::Tensor>
  butterfly_multiply_bw_cpu(const torch::Tensor twiddle,
                            const torch::Tensor input,
                            const torch::Tensor grad,
                            bool increasing_stride);
