#include "butterfly_cpu.h"
#include <type_traits>  // for std::is_floating_type

torch::Tensor butterfly_multiply_fw_cpu(const torch::Tensor& twiddle,
                                        const torch::Tensor& input,
                                        bool increasing_stride) {
  const auto batch_size = input.size(0);
  const auto nstacks = input.size(1);
  const auto n = input.size(2);
  const int log_n = int(log2((double)n));
  const int nblocks = twiddle.size(1);
  auto output =
    torch::empty({batch_size, nstacks, n},
                  torch::dtype(input.dtype()).device(input.device()));
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(input.scalar_type(), "butterfly_multiply_fw_cpu", [&] {
    const auto twiddle_a = twiddle.accessor<scalar_t, 6>();
    const auto input_a = input.accessor<scalar_t, 3>();
    auto output_a = output.accessor<scalar_t, 3>();
    for (int64_t block = 0; block < nblocks; block++) {
      bool cur_increasing_stride = increasing_stride != bool(block % 2);
      for (int64_t idx = 0; idx <= log_n - 1; ++idx) {
        auto previous_a = (idx == 0 && block == 0) ? input_a : output_a;
        int64_t log_stride = cur_increasing_stride ? idx : (log_n - 1 - idx);
        int64_t stride = 1 << log_stride;
        for (int64_t b = 0; b < batch_size; ++b) {
          for (int64_t s = 0; s < nstacks; ++s) {
            for (int64_t i = 0; i < n / 2; ++i) {
              int64_t low_order_bit = i % stride;
              int64_t pos = 2 * (i - low_order_bit) + low_order_bit;
              const scalar_t twiddle_val[2][2] =
                {{twiddle_a[s][block][idx][i][0][0], twiddle_a[s][block][idx][i][0][1]},
                 {twiddle_a[s][block][idx][i][1][0], twiddle_a[s][block][idx][i][1][1]}};
              const scalar_t input_val[2] = {previous_a[b][s][pos], previous_a[b][s][pos + stride]};
              output_a[b][s][pos] = twiddle_val[0][0] * input_val[0] + twiddle_val[0][1] * input_val[1];
              output_a[b][s][pos + stride] = twiddle_val[1][0] * input_val[0] + twiddle_val[1][1] * input_val[1];
            }
          }
        }
      }
    }
  });
  return output;
}

std::tuple<torch::Tensor, torch::Tensor>
  butterfly_multiply_bw_cpu(const torch::Tensor &twiddle,
                            const torch::Tensor &input,
                            const torch::Tensor &grad,
                            bool increasing_stride) {
  return std::make_tuple(input, grad);
}
