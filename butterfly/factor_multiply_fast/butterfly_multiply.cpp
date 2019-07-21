#include <cmath>
#include <torch/extension.h>
#include <utility>
#include <vector>

#define CHECK_DEVICE(x) AT_CHECK(x.type().device_type() == at::kCPU || x.type().device_type() == at::kCUDA, #x " must be on CPU or CUDA")

void butterfly_multiply_untied_forward_fast_cuda(const at::Tensor &twiddle,
                                                 const at::Tensor &input,
                                                 at::Tensor &output,
                                                 bool increasing_stride);
void butterfly_multiply_untied_forward_backward_fast_cuda(const at::Tensor &twiddle,
                                                          const at::Tensor &input,
                                                          const at::Tensor &grad,
                                                          at::Tensor &d_twiddle,
                                                          at::Tensor &d_input,
                                                          bool increasing_stride);

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
  AT_CHECK(n <= 1024,
           "butterfly_multiply_untied_forward_fast: only supports n <= 1024");
  const int log_n = int(log2((double)n));
  AT_CHECK((twiddle.dim() == 4 && input.dim() == 3),
           "butterfly_multiply_untied_forward_fast: twiddle and input must have "
           "dimension 4,3 or 6,4");
  CHECK_DEVICE(twiddle);
  CHECK_DEVICE(input);
  AT_CHECK(twiddle.device() == input.device(), "device of twiddle (",
           twiddle.device(), ") must match device of input (", input.device(),
           ")");
  AT_CHECK(twiddle.size(0) == nstack && twiddle.size(1) == log_n &&
               twiddle.size(2) == 2 && twiddle.size(3) == n,
           "butterfly_multiply_untied_forward_fast: twiddle must have shape (nstack, "
           "log n, 2, n) (nstack, log n, 2, n, 2)");
  auto output = torch::empty_like(input);
  AT_CHECK(input.is_cuda(), "butterfly_multiply_untied_forward_fast: only supports CUDA");
  butterfly_multiply_untied_forward_fast_cuda(twiddle, input, output, increasing_stride);
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
  AT_CHECK(n <= 1024,
           "butterfly_multiply_untied_forward_backward_fast: only supports n <= 1024");
  const int log_n = int(log2((double)n));
  AT_CHECK((twiddle.dim() == 4 && input.dim() == 3 && grad.dim() == 3),
           "butterfly_multiply_untied_forward_backward_fast: twiddle, input, "
           "and grad must have dimension 4,3,3 or 6,4,4");
  CHECK_DEVICE(twiddle);
  CHECK_DEVICE(input);
  CHECK_DEVICE(grad);
  AT_CHECK(
      twiddle.device() == input.device() && twiddle.device() == grad.device(),
      "device of twiddle (", twiddle.device(), ") must match device of input (",
      input.device(), ") and grad (", grad.device(), ")");
  AT_CHECK(twiddle.size(0) == nstack && twiddle.size(1) == log_n &&
               twiddle.size(2) == 2 && twiddle.size(3) == n,
           "butterfly_multiply_untied_forward_backward_fast: twiddle must have shape (nstack, "
           "log n, 2, n) (nstack, log n, 2, n, 2)");
  AT_CHECK(grad.size(0) == batch_size && grad.size(1) == nstack &&
               grad.size(2) == n,
           "butterfly_multiply_untied_forward_backward: grad must have shape "
           "(batch_size, nstack, n)");
  auto d_input = torch::empty_like(input);
  auto d_twiddle = torch::zeros_like(twiddle);
  AT_CHECK(input.is_cuda(), "butterfly_multiply_untied_forward_backward_fast: only supports CUDA");
  butterfly_multiply_untied_forward_backward_fast_cuda(twiddle, input, grad,
                                                       d_twiddle, d_input,
                                                       increasing_stride);
  return {d_twiddle, d_input} ;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("butterfly_multiply_untied_forward_fast",
        &butterfly_multiply_untied_forward_fast,
        "Butterfly multiply untied forward fast");
  m.def("butterfly_multiply_untied_forward_backward_fast",
        &butterfly_multiply_untied_forward_backward_fast,
        "Butterfly multiply untied forward backward fast");
}
