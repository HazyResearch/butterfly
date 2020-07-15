#include <Python.h>
#include <torch/script.h>

#include "cpu/butterfly_cpu.h"

#ifdef WITH_CUDA
#include "cuda/butterfly_cuda.h"
#endif

#ifdef _WIN32
PyMODINIT_FUNC PyInit__butterfly(void) { return NULL; }
#endif

#define CHECK_DEVICE(x) TORCH_CHECK(x.device().type() == torch::kCPU || x.device().type() == torch::kCUDA, #x " must be on CPU or CUDA")

torch::Tensor butterfly_multiply_fw(const torch::Tensor& twiddle,
                                    const torch::Tensor& input,
                                    bool increasing_stride,
                                    bool return_intermediates) {
  /* Parameters:
         twiddle: (nstack, log n, n/2, 2, 2) if real or (nstack, log n, n/2, 2, 2, 2) if complex
         input: (batch_size, nstack, n) if real or (batch_size, nstack, n, 2) if complex
         increasing_stride: whether to multiply with increasing stride (e.g. 1, 2, ..., n/2) or
             decreasing stride (e.g., n/2, n/4, ..., 1).
             Note that this only changes the order of multiplication, not how twiddle is stored.
             In other words, twiddle[@log_stride] always stores the twiddle for @stride.
         return_intermediates: whether to return just the output (i.e. computed in-place) or output
             and intermediate values for backward pass.
     Returns:
         if return_intermediates:
             output + intermediate values for backward pass: (log n + 1, batch_size, nstack, n) if real or (log n + 1, batch_size, nstack, n, 2) if complex
         else:
             output: (batch_size, nstack, n) if real or (batch_size, nstack, n, 2) if complex
  */
  const auto batch_size = input.size(0);
  const auto nstack = input.size(1);
  const auto n = input.size(2);
  const int log_n = int(log2((double) n));
  TORCH_CHECK((twiddle.dim() == 5 && input.dim() == 3) || (twiddle.dim() == 6 && input.dim() == 4),
           "butterfly_multiply_fw: twiddle and input must have dimension 5,3 or 6,4");
  CHECK_DEVICE(twiddle);
  CHECK_DEVICE(input);
  TORCH_CHECK(twiddle.device() == input.device(), "device of twiddle (", twiddle.device(), ") must match device of input (", input.device(), ")");
  TORCH_CHECK(twiddle.size(0) == nstack && twiddle.size(1) == log_n && twiddle.size(2) == n / 2 && twiddle.size(3) == 2 && twiddle.size(4) == 2,
           "butterfly_multiply_fw: twiddle must have shape (nstack, log n, n/2, 2, 2) or (nstack, log n, n/2, 2, 2, 2)");
  if (input.device().is_cuda()) {
#ifdef WITH_CUDA
    return butterfly_multiply_fw_cuda(twiddle, input, increasing_stride, return_intermediates);
#else
    TORCH_ERROR("Not compiled with CUDA support");
#endif
  } else {
    return butterfly_multiply_fw_cpu(twiddle, input, increasing_stride, return_intermediates);
  }
}
