#include "butterfly_cpu.h"

// We manually overload conj because std::conj does not work types other than
// c10::complex.
template <typename scalar_t>
static inline scalar_t conj_wrapper(scalar_t v) {
  return v;
}

template <typename T>
static inline c10::complex<T> conj_wrapper(c10::complex<T> v) {
  return std::conj(v);
}

torch::Tensor butterfly_multiply_fw_cpu(const torch::Tensor twiddle,
                                        const torch::Tensor input,
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
    for (int64_t block = 0; block < nblocks; ++block) {
      bool cur_increasing_stride = increasing_stride != bool(block % 2);
      for (int64_t idx = 0; idx < log_n; ++idx) {
        auto prev_input_a = (block == 0 && idx == 0) ? input_a : output_a;
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
              const scalar_t input_val[2] = {prev_input_a[b][s][pos], prev_input_a[b][s][pos + stride]};
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
  butterfly_multiply_bw_cpu(const torch::Tensor twiddle,
                            const torch::Tensor input,
                            const torch::Tensor grad,
                            bool increasing_stride) {
  const auto batch_size = input.size(0);
  const auto nstacks = input.size(1);
  const auto n = input.size(2);
  const int log_n = int(log2((double)n));
  const int nblocks = twiddle.size(1);
  auto output =
    torch::empty({nblocks, log_n, batch_size, nstacks, n},
                  torch::dtype(input.dtype()).device(input.device()));
  auto d_input =
      torch::empty({batch_size, nstacks, n},
                   torch::dtype(input.dtype()).device(input.device()));
  auto d_twiddle = torch::zeros_like(twiddle);
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(input.scalar_type(), "butterfly_multiply_bw_cpu", [&] {
    const auto twiddle_a = twiddle.accessor<scalar_t, 6>();
    const auto input_a = input.accessor<scalar_t, 3>();
    // Do forward pass, storing all the intermediate values
    auto output_a = output.accessor<scalar_t, 5>();
    auto prev_input_a = input_a;
    for (int64_t block = 0; block < nblocks; ++block) {
      bool cur_increasing_stride = increasing_stride != bool(block % 2);
      // Don't need the very last output for the backward pass
      for (int64_t idx = 0; idx < (block == nblocks - 1 ? log_n - 1 : log_n); ++idx) {
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
              const scalar_t input_val[2] = {prev_input_a[b][s][pos], prev_input_a[b][s][pos + stride]};
              output_a[block][idx][b][s][pos] = twiddle_val[0][0] * input_val[0] + twiddle_val[0][1] * input_val[1];
              output_a[block][idx][b][s][pos + stride] = twiddle_val[1][0] * input_val[0] + twiddle_val[1][1] * input_val[1];
            }
          }
        }
        prev_input_a = output_a[block][idx];
      }
    }
    // Backward pass
    const auto grad_a = grad.accessor<scalar_t, 3>();
    auto d_twiddle_a = d_twiddle.accessor<scalar_t, 6>();
    auto d_input_a = d_input.accessor<scalar_t, 3>();
    for (int64_t block = nblocks - 1; block >= 0; --block) {
      bool cur_increasing_stride = increasing_stride != bool(block % 2);
      for (int64_t idx = log_n - 1; idx >= 0; --idx) {
        auto prev_grad_a = (block == nblocks - 1 && idx == log_n - 1) ? grad_a : d_input_a;
        if (block == 0 && idx == 0) {
          prev_input_a = input_a;
        } else {
          prev_input_a = idx != 0 ? output_a[block][idx - 1] : output_a[block - 1][log_n - 1];
        }
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
              const scalar_t grad_val[2] = {prev_grad_a[b][s][pos], prev_grad_a[b][s][pos + stride]};
              d_input_a[b][s][pos] = conj_wrapper(twiddle_val[0][0]) * grad_val[0] + conj_wrapper(twiddle_val[1][0]) * grad_val[1];
              d_input_a[b][s][pos + stride] = conj_wrapper(twiddle_val[0][1]) * grad_val[0] + conj_wrapper(twiddle_val[1][1]) * grad_val[1];
              const scalar_t input_val[2] = {prev_input_a[b][s][pos], prev_input_a[b][s][pos + stride]};
              d_twiddle_a[s][block][idx][i][0][0] += grad_val[0] * conj_wrapper(input_val[0]);
              d_twiddle_a[s][block][idx][i][0][1] += grad_val[0] * conj_wrapper(input_val[1]);
              d_twiddle_a[s][block][idx][i][1][0] += grad_val[1] * conj_wrapper(input_val[0]);
              d_twiddle_a[s][block][idx][i][1][1] += grad_val[1] * conj_wrapper(input_val[1]);
            }
          }
        }
      }
    }
  });
  return std::make_tuple(d_twiddle, d_input);
}
