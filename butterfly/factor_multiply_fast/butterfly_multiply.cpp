#include <cmath>
#include <torch/extension.h>
#include <utility>
#include <vector>

#define BFLY_BENCHMARK false

#define CHECK_DEVICE(x) TORCH_CHECK(x.type().device_type() == at::kCPU || x.type().device_type() == at::kCUDA, #x " must be on CPU or CUDA")

void butterfly_multiply_untied_forward_fast_cuda(const at::Tensor &twiddle,
                                                 const at::Tensor &input,
                                                 at::Tensor &output,
                                                 bool increasing_stride);
#if BFLY_BENCHMARK
void butterfly_multiply_untied_forward_fast_cuda_benchmark(const at::Tensor &twiddle,
                                                           const at::Tensor &input,
                                                           at::Tensor &output,
                                                           bool increasing_stride);
#endif
void butterfly_multiply_untied_forward_backward_fast_cuda(const at::Tensor &twiddle,
                                                          const at::Tensor &input,
                                                          const at::Tensor &grad,
                                                          at::Tensor &d_twiddle,
                                                          at::Tensor &d_input,
                                                          bool increasing_stride);
void butterfly_bbs_multiply_untied_forward_fast_cuda(const at::Tensor &twiddle,
                                                     const at::Tensor &input,
                                                     at::Tensor &output);
void butterfly_bbs_multiply_untied_forward_backward_fast_cuda(const at::Tensor &twiddle,
                                                              const at::Tensor &input,
                                                              const at::Tensor &grad,
                                                              at::Tensor &d_twiddle,
                                                              at::Tensor &d_input);
void butterfly_ortho_multiply_untied_forward_fast_cuda(const at::Tensor &twiddle_cos,
                                                       const at::Tensor &twiddle_sin,
                                                       const at::Tensor &input,
                                                       at::Tensor &output,
                                                       bool increasing_stride);
void butterfly_ortho_multiply_untied_backward_fast_cuda(const at::Tensor &twiddle_cos,
                                                        const at::Tensor &twiddle_sin,
                                                        const at::Tensor &output,
                                                        const at::Tensor &grad,
                                                        at::Tensor &d_twiddle,
                                                        at::Tensor &d_input,
                                                        bool increasing_stride);
void butterfly_odo_multiply_untied_forward_fast_cuda(const at::Tensor &twiddle_cos,
                                                     const at::Tensor &twiddle_sin,
                                                     const at::Tensor &diagonal,
                                                     const at::Tensor &input,
                                                     at::Tensor &output);
#if BFLY_BENCHMARK
void butterfly_odo_multiply_untied_forward_fast_cuda_benchmark(const at::Tensor &twiddle_cos,
                                                               const at::Tensor &twiddle_sin,
                                                               const at::Tensor &diagonal,
                                                               const at::Tensor &input,
                                                               at::Tensor &output);
#endif
void butterfly_odo_multiply_untied_backward_fast_cuda(const at::Tensor &twiddle_cos,
                                                      const at::Tensor &twiddle_sin,
                                                      const at::Tensor &diagonal,
                                                      const at::Tensor &output,
                                                      const at::Tensor &grad,
                                                      at::Tensor &d_twiddle,
                                                      at::Tensor &d_diagonal,
                                                      at::Tensor &d_input);
#if BFLY_BENCHMARK
void butterfly_odo_multiply_untied_backward_fast_cuda_benchmark(const at::Tensor &twiddle_cos,
                                                                const at::Tensor &twiddle_sin,
                                                                const at::Tensor &diagonal,
                                                                const at::Tensor &output,
                                                                const at::Tensor &grad,
                                                                at::Tensor &d_twiddle,
                                                                at::Tensor &d_diagonal,
                                                                at::Tensor &d_input);
#endif
void butterfly_odo_multiply_untied_forward_backward_fast_cuda(const at::Tensor &twiddle_cos,
                                                              const at::Tensor &twiddle_sin,
                                                              const at::Tensor &diagonal,
                                                              const at::Tensor &input,
                                                              const at::Tensor &grad,
                                                              at::Tensor &d_twiddle,
                                                              at::Tensor &d_diagonal,
                                                              at::Tensor &d_input);
#if BFLY_BENCHMARK
void butterfly_odo_multiply_untied_forward_backward_fast_cuda_benchmark(const at::Tensor &twiddle_cos,
                                                                        const at::Tensor &twiddle_sin,
                                                                        const at::Tensor &diagonal,
                                                                        const at::Tensor &input,
                                                                        const at::Tensor &grad,
                                                                        at::Tensor &d_twiddle,
                                                                        at::Tensor &d_diagonal,
                                                                        at::Tensor &d_input);
#endif

at::Tensor butterfly_multiply_untied_forward_fast(const at::Tensor &twiddle,
                                                  const at::Tensor &input,
                                                  bool increasing_stride) {
  /* Parameters:
         twiddle: (nstack, log n, n/2, 2, 2) if real or (nstack, log n, n/2, 2, 2, 2) if complex
         input: (batch_size, nstack, n) if real or (batch_size, nstack, n, 2) if complex
         increasing_stride: whether to multiply with increasing stride (e.g. 1, 2, ..., n/2) or
             decreasing stride (e.g., n/2, n/4, ..., 1).
     Returns:
         output: (batch_size, nstack, n) if real or (batch_size, nstack, n, 2) if complex
  */
  // const auto batch_size = input.size(0);
  const auto nstack = input.size(1);
  const auto n = input.size(2);
  const int log_n = int(log2((double)n));
  TORCH_CHECK(1 << log_n == n, "butterfly_multiply_untied_forward_fast: n must be a power of 2");
  TORCH_CHECK(n <= 16384,
           "butterfly_multiply_untied_forward_fast: only supports n <= 16384");
  TORCH_CHECK((twiddle.dim() == 4 && input.dim() == 3),
           "butterfly_multiply_untied_forward_fast: twiddle and input must have "
           "dimension 4,3 or 6,4");
  CHECK_DEVICE(twiddle);
  CHECK_DEVICE(input);
  TORCH_CHECK(twiddle.device() == input.device(), "device of twiddle (",
           twiddle.device(), ") must match device of input (", input.device(),
           ")");
  TORCH_CHECK(twiddle.sizes() == torch::IntArrayRef({nstack, log_n, 2, n}),
           "butterfly_multiply_untied_forward_fast: twiddle must have shape (nstack, "
           "log n, 2, n) (nstack, log n, 2, n, 2)");
  auto output = torch::empty_like(input);
  TORCH_CHECK(input.is_cuda(), "butterfly_multiply_untied_forward_fast: only supports CUDA");
  #if !BFLY_BENCHMARK
  butterfly_multiply_untied_forward_fast_cuda(twiddle, input, output, increasing_stride);
  #else
  butterfly_multiply_untied_forward_fast_cuda_benchmark(twiddle, input, output, increasing_stride);
  #endif
  return output;
}

std::vector<at::Tensor> butterfly_multiply_untied_forward_backward_fast(const at::Tensor &twiddle,
                                                                        const at::Tensor &input,
                                                                        const at::Tensor &grad,
                                                                        bool increasing_stride) {
  /* Parameters:
         twiddle: (nstack, log n, n/2, 2, 2) if real or (nstack, log n, n/2, 2, 2, 2) if complex
         input: (batch_size, nstack, n) if real or (batch_size, nstack, n, 2) if complex
         increasing_stride: whether to multiply with increasing stride (e.g. 1, 2, ..., n/2) or
             decreasing stride (e.g., n/2, n/4, ..., 1).
     Returns:
         output: (batch_size, nstack, n) if real or (batch_size, nstack, n, 2) if complex
  */
  const auto batch_size = input.size(0);
  const auto nstack = input.size(1);
  const auto n = input.size(2);
  const int log_n = int(log2((double)n));
  TORCH_CHECK(1 << log_n == n, "butterfly_multiply_untied_forward_backward_fast: n must be a power of 2");
  TORCH_CHECK(n <= 4096,
           "butterfly_multiply_untied_forward_backward_fast: only supports n <= 4096");
  TORCH_CHECK((twiddle.dim() == 4 && input.dim() == 3 && grad.dim() == 3),
           "butterfly_multiply_untied_forward_backward_fast: twiddle, input, "
           "and grad must have dimension 4,3,3");
  CHECK_DEVICE(twiddle);
  CHECK_DEVICE(input);
  CHECK_DEVICE(grad);
  TORCH_CHECK(
      twiddle.device() == input.device() && twiddle.device() == grad.device(),
      "device of twiddle (", twiddle.device(), ") must match device of input (",
      input.device(), ") and grad (", grad.device(), ")");
  TORCH_CHECK(twiddle.sizes() == torch::IntArrayRef({nstack, log_n, 2, n}),
           "butterfly_multiply_untied_forward_backward_fast: twiddle must have shape (nstack, "
           "log n, 2, n) (nstack, log n, 2, n, 2)");
  TORCH_CHECK(grad.sizes() == torch::IntArrayRef({batch_size, nstack, n}),
           "butterfly_multiply_untied_forward_backward: grad must have shape "
           "(batch_size, nstack, n)");
  auto d_input = torch::empty_like(input);
  auto d_twiddle = torch::zeros_like(twiddle);
  TORCH_CHECK(input.is_cuda(), "butterfly_multiply_untied_forward_backward_fast: only supports CUDA");
  butterfly_multiply_untied_forward_backward_fast_cuda(twiddle, input, grad,
                                                       d_twiddle, d_input,
                                                       increasing_stride);
  return {d_twiddle, d_input} ;
}

at::Tensor butterfly_bbs_multiply_untied_forward_fast(const at::Tensor &twiddle,
                                                      const at::Tensor &input) {
  /* Parameters:
         twiddle: (nstack, nblocks * 2 * log n, 2, n)
         input: (batch_size, nstack, n)
     Returns:
         output: (batch_size, nstack, n)
  */
  const auto nstack = input.size(1);
  const auto n = input.size(2);
  const int log_n = int(log2((double)n));
  TORCH_CHECK(1 << log_n == n, "butterfly_bbs_multiply_untied_forward_fast: n must be a power of 2");
  TORCH_CHECK(n <= 16384,
           "butterfly_bbs_multiply_untied_forward_fast: only supports n <= 16384");
  const int nblocks = twiddle.size(1) / (2 * log_n);
  TORCH_CHECK((twiddle.dim() == 4 && input.dim() == 3),
           "butterfly_bbs_multiply_untied_forward_fast: twiddle and input must have "
           "dimension 3,3");
  CHECK_DEVICE(twiddle);
  CHECK_DEVICE(input);
  TORCH_CHECK(twiddle.device() == input.device(),
           "device of twiddle (", twiddle.device(), ") must match device of input (", input.device(),
           ")");
  TORCH_CHECK(twiddle.sizes() ==
               torch::IntArrayRef({nstack, nblocks * 2 * log_n, 2, n}),
           "butterfly_bbs_multiply_untied_forward_fast: twiddle must have "
           "shape (nstack, nblocks * 2 * log n, 2, n)");
  auto output = torch::empty_like(input);
  TORCH_CHECK(input.is_cuda(), "butterfly_bbs_multiply_untied_forward_fast: only supports CUDA");
  butterfly_bbs_multiply_untied_forward_fast_cuda(twiddle, input, output);
  return output;
}

std::vector<at::Tensor> butterfly_bbs_multiply_untied_forward_backward_fast(const at::Tensor &twiddle,
                                                                            const at::Tensor &input,
                                                                            const at::Tensor &grad) {
  /* Parameters:
         twiddle: (nstack, nblocks * 2 * log n, 2, n)
         diagonal: (nstack, nblocks, n)
         input: (batch_size, nstack, n)
     Returns:
         input: (batch_size, nstack, n)
  */
  const auto batch_size = input.size(0);
  const auto nstack = input.size(1);
  const auto n = input.size(2);
  const int log_n = int(log2((double)n));
  TORCH_CHECK(1 << log_n == n, "butterfly_bbs_multiply_untied_forward_backward_fast: n must be a power of 2");
  TORCH_CHECK(n <= 4096,
           "butterfly_bbs_multiply_untied_forward_backward_fast: only supports n <= 4096");
  const int nblocks = twiddle.size(1) / (2 * log_n);
  TORCH_CHECK((twiddle.dim() == 4 && input.dim() == 3 && grad.dim() == 3),
           "butterfly_bbs_multiply_untied_forward_backward_fast: twiddle, input, "
           "and grad must have dimension 4,3,3");
  CHECK_DEVICE(twiddle);
  CHECK_DEVICE(input);
  CHECK_DEVICE(grad);
  TORCH_CHECK(
      twiddle.device() == input.device() && twiddle.device() == grad.device(),
      "device of twiddle (", twiddle.device(), ") must match device of input (",
      input.device(), ") and grad (", grad.device(), ")");
  TORCH_CHECK(twiddle.sizes() == torch::IntArrayRef({nstack, nblocks * 2 * log_n, 2, n}),
           "butterfly_bbs_multiply_untied_forward_backward_fast: twiddle must have shape (nstack, "
           "nblocks * 2 * log n, 2, n)");
  TORCH_CHECK(grad.sizes() == torch::IntArrayRef({batch_size, nstack, n}),
           "butterfly_bbs_multiply_untied_backward: grad must have shape "
           "(batch_size, nstack, n)");
  auto d_input = torch::empty_like(input);
  auto d_twiddle = torch::zeros_like(twiddle);
  TORCH_CHECK(input.is_cuda(), "butterfly_bbs_multiply_untied_forward_backward_fast: only supports CUDA");
  butterfly_bbs_multiply_untied_forward_backward_fast_cuda(twiddle, input, grad,
                                                           d_twiddle, d_input);
  return {d_twiddle, d_input} ;
}

at::Tensor butterfly_ortho_multiply_untied_forward_fast(const at::Tensor &twiddle_cos,
                                                        const at::Tensor &twiddle_sin,
                                                        const at::Tensor &input,
                                                        bool increasing_stride) {
  /* Parameters:
         twiddle_cos: (nstack, log n, n/2)
         twiddle_sin: (nstack, log n, n/2)
         input: (batch_size, nstack, n)
         increasing_stride: whether to multiply with increasing stride (e.g. 1, 2, ..., n/2) or
             decreasing stride (e.g., n/2, n/4, ..., 1).
     Returns:
         output: (batch_size, nstack, n)
  */
  const auto nstack = input.size(1);
  const auto n = input.size(2);
  const int log_n = int(log2((double)n));
  TORCH_CHECK(1 << log_n == n, "butterfly_ortho_multiply_untied_forward_fast: n must be a power of 2");
  TORCH_CHECK(n <= 16384,
           "butterfly_ortho_multiply_untied_forward_fast: only supports n <= 16384");
  TORCH_CHECK((twiddle_cos.dim() == 3 && twiddle_sin.dim() == 3 && input.dim() == 3),
           "butterfly_ortho_multiply_untied_forward_fast: twiddle_cos, twiddle_sin and input must have "
           "dimension 3,3,3");
  CHECK_DEVICE(twiddle_cos);
  CHECK_DEVICE(twiddle_sin);
  CHECK_DEVICE(input);
  TORCH_CHECK(twiddle_cos.device() == input.device() && twiddle_sin.device() == input.device(),
           "device of twiddle_cos (", twiddle_cos.device(), ") must match device of input (", input.device(),
           ")");
  TORCH_CHECK(twiddle_cos.sizes() == torch::IntArrayRef({nstack, log_n, n / 2}),
           "butterfly_ortho_multiply_untied_forward_fast: twiddle_cos must have shape (nstack, "
           "log n, n/2)");
  TORCH_CHECK(twiddle_sin.sizes() == torch::IntArrayRef({nstack, log_n, n / 2}),
           "butterfly_ortho_multiply_untied_forward_fast: twiddle_sin must have shape (nstack, "
           "log n, n/2)");
  auto output = torch::empty_like(input);
  TORCH_CHECK(input.is_cuda(), "butterfly_ortho_multiply_untied_forward_fast: only supports CUDA");
  butterfly_ortho_multiply_untied_forward_fast_cuda(twiddle_cos, twiddle_sin, input, output, increasing_stride);
  return output;
}

std::vector<at::Tensor> butterfly_ortho_multiply_untied_backward_fast(const at::Tensor &twiddle_cos,
                                                                      const at::Tensor &twiddle_sin,
                                                                      const at::Tensor &output,
                                                                      const at::Tensor &grad,
                                                                      bool increasing_stride) {
  /* Parameters:
         twiddle_cos: (nstack, log n, n/2)
         twiddle_sin: (nstack, log n, n/2)
         output: (batch_size, nstack, n)
         increasing_stride: whether to multiply with increasing stride (e.g. 1, 2, ..., n/2) or
             decreasing stride (e.g., n/2, n/4, ..., 1).
     Returns:
         output: (batch_size, nstack, n)
  */
  const auto batch_size = output.size(0);
  const auto nstack = output.size(1);
  const auto n = output.size(2);
  const int log_n = int(log2((double)n));
  TORCH_CHECK(1 << log_n == n, "butterfly_ortho_multiply_untied_backward_fast: n must be a power of 2");
  TORCH_CHECK(n <= 16384,
           "butterfly_ortho_multiply_untied_backward_fast: only supports n <= 4096");
  TORCH_CHECK((twiddle_cos.dim() == 3 && twiddle_sin.dim() == 3 && output.dim() == 3 && grad.dim() == 3),
           "butterfly_ortho_multiply_untied_backward_fast: twiddle_cos, twiddle_sin, output, "
           "and grad must have dimension 3,3,3,3");
  CHECK_DEVICE(twiddle_cos);
  CHECK_DEVICE(twiddle_sin);
  CHECK_DEVICE(output);
  CHECK_DEVICE(grad);
  TORCH_CHECK(
      twiddle_cos.device() == output.device() && twiddle_cos.device() == grad.device()
      && twiddle_cos.device() == twiddle_sin.device(),
      "device of twiddle_cos (", twiddle_cos.device(), ") must match device of output (",
      output.device(), ") and grad (", grad.device(), ")");
  TORCH_CHECK(twiddle_cos.sizes() == torch::IntArrayRef({nstack, log_n, n / 2}),
           "butterfly_ortho_multiply_untied_backward_fast: twiddle_cos must have shape (nstack, "
           "log n, n / 2)");
  TORCH_CHECK(twiddle_sin.sizes() == torch::IntArrayRef({nstack, log_n, n / 2}),
           "butterfly_ortho_multiply_untied_backward_fast: twiddle_sin must have shape (nstack, "
           "log n, n / 2)");
  TORCH_CHECK(grad.sizes() == torch::IntArrayRef({batch_size, nstack, n}),
           "butterfly_ortho_multiply_untied_backward: grad must have shape "
           "(batch_size, nstack, n)");
  auto d_input = torch::empty_like(output);
  auto d_twiddle = torch::zeros_like(twiddle_cos);
  TORCH_CHECK(output.is_cuda(), "butterfly_ortho_multiply_untied_backward_fast: only supports CUDA");
  butterfly_ortho_multiply_untied_backward_fast_cuda(twiddle_cos, twiddle_sin, output, grad,
                                                     d_twiddle, d_input, increasing_stride);
  return {d_twiddle, d_input} ;
}

at::Tensor butterfly_odo_multiply_untied_forward_fast(const at::Tensor &twiddle_cos,
                                                      const at::Tensor &twiddle_sin,
                                                      const at::Tensor &diagonal,
                                                      const at::Tensor &input) {
  /* Parameters:
         twiddle_cos: (nstack, nblocks * 2 * log n, n/2)
         twiddle_sin: (nstack, nblocks * 2 * log n, n/2)
         diagonal: (nstack, nblocks, n)
         input: (batch_size, nstack, n)
     Returns:
         output: (batch_size, nstack, n)
  */
  const auto nstack = input.size(1);
  const auto n = input.size(2);
  const int log_n = int(log2((double)n));
  TORCH_CHECK(1 << log_n == n, "butterfly_odo_multiply_untied_forward_fast: n must be a power of 2");
  TORCH_CHECK(n <= 16384,
           "butterfly_odo_multiply_untied_forward_fast: only supports n <= 16384");
  const int nblocks = twiddle_cos.size(1) / (2 * log_n);
  TORCH_CHECK((twiddle_cos.dim() == 3 && twiddle_sin.dim() == 3 && diagonal.dim() == 3 && input.dim() == 3),
           "butterfly_odo_multiply_untied_forward_fast: twiddle_cos, twiddle_sin, diagonal and input must have "
           "dimension 3,3,3,3");
  CHECK_DEVICE(twiddle_cos);
  CHECK_DEVICE(twiddle_sin);
  CHECK_DEVICE(diagonal);
  CHECK_DEVICE(input);
  TORCH_CHECK(twiddle_cos.device() == input.device() && twiddle_sin.device() == input.device() && diagonal.device() == input.device(),
           "device of twiddle_cos (", twiddle_cos.device(), ") must match device of input (", input.device(),
           ")");
  TORCH_CHECK(twiddle_cos.sizes() == torch::IntArrayRef({nstack, nblocks * 2 * log_n, n / 2}),
           "butterfly_odo_multiply_untied_forward_fast: twiddle_cos must have shape (nstack, "
           "nblocks * 2 * log n, n/2)");
  TORCH_CHECK(twiddle_sin.sizes() == torch::IntArrayRef({nstack, nblocks * 2 * log_n, n / 2}),
           "butterfly_odo_multiply_untied_forward_fast: twiddle_sin must have shape (nstack, "
           "nblocks * 2 * log n, n/2)");
  TORCH_CHECK(diagonal.sizes() == torch::IntArrayRef({nstack, nblocks, n}),
           "butterfly_odo_multiply_untied_forward_fast: diagonal must have shape (nstack, "
           "nblocks, n/2)");
  auto output = torch::empty_like(input);
  TORCH_CHECK(input.is_cuda(), "butterfly_odo_multiply_untied_forward_fast: only supports CUDA");
  #if !BFLY_BENCHMARK
  butterfly_odo_multiply_untied_forward_fast_cuda(twiddle_cos, twiddle_sin, diagonal, input, output);
  #else
  butterfly_odo_multiply_untied_forward_fast_cuda_benchmark(twiddle_cos, twiddle_sin, diagonal, input, output);
  #endif
  return output;
}

std::vector<at::Tensor> butterfly_odo_multiply_untied_backward_fast(const at::Tensor &twiddle_cos,
                                                                    const at::Tensor &twiddle_sin,
                                                                    const at::Tensor &diagonal,
                                                                    const at::Tensor &output,
                                                                    const at::Tensor &grad) {
  /* Parameters:
         twiddle_cos: (nstack, nblocks * 2 * log n, n/2)
         twiddle_sin: (nstack, nblocks * 2 * log n, n/2)
         diagonal: (nstack, nblocks, n)
         output: (batch_size, nstack, n)
     Returns:
         output: (batch_size, nstack, n)
  */
  const auto batch_size = output.size(0);
  const auto nstack = output.size(1);
  const auto n = output.size(2);
  const int log_n = int(log2((double)n));
  TORCH_CHECK(1 << log_n == n, "butterfly_odo_multiply_untied_backward_fast: n must be a power of 2");
  TORCH_CHECK(n <= 16384,
           "butterfly_odo_multiply_untied_backward_fast: only supports n <= 4096");
  const int nblocks = twiddle_cos.size(1) / (2 * log_n);
  TORCH_CHECK((twiddle_cos.dim() == 3 && twiddle_sin.dim() == 3 && diagonal.dim() == 3 && output.dim() == 3 && grad.dim() == 3),
           "butterfly_odo_multiply_untied_backward_fast: twiddle_cos, twiddle_sin, diagonal, output, "
           "and grad must have dimension 3,3,3,3");
  CHECK_DEVICE(twiddle_cos);
  CHECK_DEVICE(twiddle_sin);
  CHECK_DEVICE(diagonal);
  CHECK_DEVICE(output);
  CHECK_DEVICE(grad);
  TORCH_CHECK(
      twiddle_cos.device() == output.device() && twiddle_cos.device() == grad.device()
      && twiddle_cos.device() == twiddle_sin.device() && twiddle_cos.device() == diagonal.device(),
      "device of twiddle_cos (", twiddle_cos.device(), ") must match device of output (",
      output.device(), ") and grad (", grad.device(), ")");
  TORCH_CHECK(twiddle_cos.sizes() == torch::IntArrayRef({nstack, nblocks * 2 * log_n, n / 2}),
           "butterfly_odo_multiply_untied_backward_fast: twiddle_cos must have shape (nstack, "
           "log n, n / 2)");
  TORCH_CHECK(twiddle_sin.sizes() == torch::IntArrayRef({nstack, nblocks * 2 * log_n, n / 2}),
           "butterfly_odo_multiply_untied_backward_fast: twiddle_sin must have shape (nstack, "
           "log n, n / 2)");
  TORCH_CHECK(diagonal.sizes() == torch::IntArrayRef({nstack, nblocks, n}),
           "butterfly_odo_multiply_untied_backward_fast: diagonal must have shape (nstack, "
           "log n, n / 2)");
  TORCH_CHECK(grad.sizes() == torch::IntArrayRef({batch_size, nstack, n}),
           "butterfly_odo_multiply_untied_backward: grad must have shape "
           "(batch_size, nstack, n)");
  auto d_input = torch::empty_like(output);
  auto d_twiddle = torch::zeros_like(twiddle_cos);
  auto d_diagonal = torch::zeros_like(diagonal);
  TORCH_CHECK(output.is_cuda(), "butterfly_odo_multiply_untied_backward_fast: only supports CUDA");
  #if !BFLY_BENCHMARK
  butterfly_odo_multiply_untied_backward_fast_cuda(twiddle_cos, twiddle_sin, diagonal, output, grad,
                                                   d_twiddle, d_diagonal, d_input);
  #else
  butterfly_odo_multiply_untied_backward_fast_cuda_benchmark(twiddle_cos, twiddle_sin, diagonal, output, grad,
                                                             d_twiddle, d_diagonal, d_input);
  #endif
  return {d_twiddle, d_diagonal, d_input} ;
}

std::vector<at::Tensor> butterfly_odo_multiply_untied_forward_backward_fast(const at::Tensor &twiddle_cos,
                                                                            const at::Tensor &twiddle_sin,
                                                                            const at::Tensor &diagonal,
                                                                            const at::Tensor &input,
                                                                            const at::Tensor &grad) {
  /* Parameters:
         twiddle_cos: (nstack, nblocks * 2 * log n, n/2)
         twiddle_sin: (nstack, nblocks * 2 * log n, n/2)
         diagonal: (nstack, nblocks, n)
         input: (batch_size, nstack, n)
     Returns:
         input: (batch_size, nstack, n)
  */
  const auto batch_size = input.size(0);
  const auto nstack = input.size(1);
  const auto n = input.size(2);
  const int log_n = int(log2((double)n));
  TORCH_CHECK(1 << log_n == n, "butterfly_odo_multiply_untied_forward_backward_fast: n must be a power of 2");
  TORCH_CHECK(n <= 16384,
           "butterfly_odo_multiply_untied_forward_backward_fast: only supports n <= 4096");
  const int nblocks = twiddle_cos.size(1) / (2 * log_n);
  TORCH_CHECK((twiddle_cos.dim() == 3 && twiddle_sin.dim() == 3 && diagonal.dim() == 3 && input.dim() == 3 && grad.dim() == 3),
           "butterfly_odo_multiply_untied_forward_backward_fast: twiddle_cos, twiddle_sin, diagonal, input, "
           "and grad must have dimension 3,3,3,3");
  CHECK_DEVICE(twiddle_cos);
  CHECK_DEVICE(twiddle_sin);
  CHECK_DEVICE(diagonal);
  CHECK_DEVICE(input);
  CHECK_DEVICE(grad);
  TORCH_CHECK(
      twiddle_cos.device() == input.device() && twiddle_cos.device() == grad.device()
      && twiddle_cos.device() == twiddle_sin.device() && twiddle_cos.device() == diagonal.device(),
      "device of twiddle_cos (", twiddle_cos.device(), ") must match device of input (",
      input.device(), ") and grad (", grad.device(), ")");
  TORCH_CHECK(twiddle_cos.sizes() == torch::IntArrayRef({nstack, nblocks * 2 * log_n, n / 2}),
           "butterfly_odo_multiply_untied_forward_backward_fast: twiddle_cos must have shape (nstack, "
           "log n, n / 2)");
  TORCH_CHECK(twiddle_sin.sizes() == torch::IntArrayRef({nstack, nblocks * 2 * log_n, n / 2}),
           "butterfly_odo_multiply_untied_forward_backward_fast: twiddle_sin must have shape (nstack, "
           "log n, n / 2)");
  TORCH_CHECK(diagonal.sizes() == torch::IntArrayRef({nstack, nblocks, n}),
           "butterfly_odo_multiply_untied_forward_backward_fast: diagonal must have shape (nstack, "
           "log n, n / 2)");
  TORCH_CHECK(grad.sizes() == torch::IntArrayRef({batch_size, nstack, n}),
           "butterfly_odo_multiply_untied_forward_backward: grad must have shape "
           "(batch_size, nstack, n)");
  auto d_input = torch::empty_like(input);
  auto d_twiddle = torch::zeros_like(twiddle_cos);
  auto d_diagonal = torch::zeros_like(diagonal);
  TORCH_CHECK(input.is_cuda(), "butterfly_odo_multiply_untied_forward_backward_fast: only supports CUDA");
  #if !BFLY_BENCHMARK
  butterfly_odo_multiply_untied_forward_backward_fast_cuda(twiddle_cos, twiddle_sin, diagonal, input, grad,
                                                           d_twiddle, d_diagonal, d_input);
  #else
  butterfly_odo_multiply_untied_forward_backward_fast_cuda_benchmark(twiddle_cos, twiddle_sin, diagonal, input, grad,
                                                                     d_twiddle, d_diagonal, d_input);
  #endif
  return {d_twiddle, d_diagonal, d_input} ;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("butterfly_multiply_untied_forward_fast",
        &butterfly_multiply_untied_forward_fast,
        "Butterfly multiply untied forward fast");
  m.def("butterfly_multiply_untied_forward_backward_fast",
        &butterfly_multiply_untied_forward_backward_fast,
        "Butterfly multiply untied forward backward fast");
  m.def("butterfly_bbs_multiply_untied_forward_fast",
        &butterfly_bbs_multiply_untied_forward_fast,
        "Butterfly_Bbs multiply untied forward fast");
  m.def("butterfly_bbs_multiply_untied_forward_backward_fast",
        &butterfly_bbs_multiply_untied_forward_backward_fast,
        "Butterfly_Bbs multiply untied forward_backward fast");
  m.def("butterfly_ortho_multiply_untied_forward_fast",
        &butterfly_ortho_multiply_untied_forward_fast,
        "Butterfly_Ortho multiply untied forward fast");
  m.def("butterfly_ortho_multiply_untied_backward_fast",
        &butterfly_ortho_multiply_untied_backward_fast,
        "Butterfly_Ortho multiply untied backward fast");
  m.def("butterfly_odo_multiply_untied_forward_fast",
        &butterfly_odo_multiply_untied_forward_fast,
        "Butterfly_Odo multiply untied forward fast");
  m.def("butterfly_odo_multiply_untied_backward_fast",
        &butterfly_odo_multiply_untied_backward_fast,
        "Butterfly_Odo multiply untied backward fast");
  m.def("butterfly_odo_multiply_untied_forward_backward_fast",
        &butterfly_odo_multiply_untied_forward_backward_fast,
        "Butterfly_Odo multiply untied forward_backward fast");
}
