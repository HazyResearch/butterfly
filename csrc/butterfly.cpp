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
#define CHECK_SAME_DEVICE(x, y) TORCH_CHECK(x.device() == y.device(), #x " and " #y " must be on the same device")

torch::Tensor butterfly_multiply_fw(const torch::Tensor& twiddle,
                                    const torch::Tensor& input,
                                    bool increasing_stride) {
  /* Parameters:
         twiddle: (nstack, log n, n/2, 2, 2)
         input: (batch_size, nstack, n)
         increasing_stride: whether to multiply with increasing stride (e.g. 1, 2, ..., n/2) or
             decreasing stride (e.g., n/2, n/4, ..., 1).
             Note that this only changes the order of multiplication, not how twiddle is stored.
             In other words, twiddle[@log_stride] always stores the twiddle for @stride.
     Returns:
         output: (batch_size, nstack, n)
  */
  CHECK_DEVICE(twiddle); CHECK_DEVICE(input); CHECK_SAME_DEVICE(twiddle, input);
  // const auto batch_size = input.size(0);
  const auto nstack = input.size(1);
  const auto n = input.size(2);
  const int log_n = int(log2((double) n));
  TORCH_CHECK((twiddle.dim() == 5 && input.dim() == 3),
              "butterfly_multiply_fw: twiddle and input must have dimension 5, 3");
  TORCH_CHECK(twiddle.sizes() == torch::IntArrayRef({nstack, log_n, n / 2, 2, 2}),
              "butterfly_multiply_fw: twiddle must have shape (nstack, log n, n/2, 2, 2)");
  if (input.device().is_cuda()) {
#ifdef WITH_CUDA
    return butterfly_multiply_fw_cuda(twiddle, input, increasing_stride);
#else
    TORCH_ERROR("Not compiled with CUDA support");
#endif
  } else {
    return butterfly_multiply_fw_cpu(twiddle, input, increasing_stride);
  }
}

static auto registry = torch::RegisterOperators().op(
    "torch_butterfly::butterfly_multiply_fw", &butterfly_multiply_fw);
