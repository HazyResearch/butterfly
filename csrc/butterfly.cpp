#include <Python.h>
#include <torch/script.h>
#include <tuple>

#include "cpu/butterfly_cpu.h"

#ifdef WITH_CUDA
#include "cuda/butterfly_cuda.h"
#endif

#ifdef _WIN32
PyMODINIT_FUNC PyInit__butterfly(void) { return NULL; }
#endif

#define CHECK_DEVICE(x) TORCH_CHECK(x.device().type() == torch::kCPU || x.device().type() == torch::kCUDA, #x " must be on CPU or CUDA")
#define CHECK_SAME_DEVICE(x, y) TORCH_CHECK(x.device() == y.device(), #x " and " #y " must be on the same device")
#define CHECK_DIM(x, y) TORCH_CHECK(x.dim() == y, #x " must have dimension " #y)
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")

torch::Tensor butterfly_multiply_fw(const torch::Tensor twiddle,
                                    const torch::Tensor input,
                                    bool increasing_stride) {
  /* Parameters:
         twiddle: (nstacks, nblocks, log n, n/2, 2, 2)
         input: (batch_size, nstacks, n)
         increasing_stride: whether the first block multiplies with increasing
             stride (e.g. 1, 2, ..., n/2) or decreasing stride (e.g., n/2, n/4, ..., 1).
     Returns:
         output: (batch_size, nstacks, n)
  */
  CHECK_DEVICE(twiddle); CHECK_DEVICE(input); CHECK_SAME_DEVICE(twiddle, input);
  const auto nstacks = input.size(1);
  const auto n = input.size(2);
  const int log_n = int(log2((double) n));
  const auto nblocks = twiddle.size(1);
  CHECK_DIM(twiddle, 6); CHECK_DIM(input, 3);
  CHECK_SHAPE(twiddle, nstacks, nblocks, log_n, n / 2, 2, 2);
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

// Has to be tuple and not pair, Pytorch doesn't like pair
std::tuple<torch::Tensor, torch::Tensor>
  butterfly_multiply_bw(const torch::Tensor twiddle,
                        const torch::Tensor input,
                        const torch::Tensor grad,
                        bool increasing_stride) {
  /* Parameters:
         twiddle: (nstacks, nblocks, log n, n/2, 2, 2)
         input: (batch_size, nstacks, n)
         grad: (batch_size, nstacks, n)
         increasing_stride: whether the first block multiplies with increasing
             stride (e.g. 1, 2, ..., n/2) or decreasing stride (e.g., n/2, n/4, ..., 1).
     Returns:
         d_twiddle: (nstacks, nblocks, log n, n/2, 2, 2)
         d_input: (batch_size, nstacks, n)
  */
  CHECK_DEVICE(twiddle); CHECK_DEVICE(input); CHECK_DEVICE(grad);
  CHECK_SAME_DEVICE(twiddle, input); CHECK_SAME_DEVICE(input, grad);
  const auto batch_size = input.size(0);
  const auto nstacks = input.size(1);
  const auto n = input.size(2);
  const int log_n = int(log2((double) n));
  const auto nblocks = twiddle.size(1);
  CHECK_DIM(twiddle, 6); CHECK_DIM(input, 3); CHECK_DIM(grad, 3);
  CHECK_SHAPE(twiddle, nstacks, nblocks, log_n, n / 2, 2, 2);
  CHECK_SHAPE(grad, batch_size, nstacks, n);
  if (input.device().is_cuda()) {
#ifdef WITH_CUDA
    return butterfly_multiply_bw_cuda(twiddle, input, grad, increasing_stride);
#else
    TORCH_ERROR("Not compiled with CUDA support");
#endif
  } else {
    return butterfly_multiply_bw_cpu(twiddle, input, grad, increasing_stride);
  }
}

using torch::autograd::AutogradContext;
using torch::autograd::variable_list;

// https://github.com/pytorch/pytorch/blob/master/test/custom_operator/op.cpp
class ButterflyMultiply : public torch::autograd::Function<ButterflyMultiply> {
public:
  static torch::Tensor forward(AutogradContext *ctx, torch::Tensor twiddle,
                               torch::Tensor input, bool increasing_stride) {
    ctx->saved_data["increasing_stride"] = increasing_stride;
    ctx->save_for_backward({twiddle, input});
    return butterfly_multiply_fw(twiddle, input, increasing_stride);
  }

  static variable_list backward(AutogradContext *ctx, variable_list grad_output) {
    auto grad = grad_output[0];
    auto saved = ctx->get_saved_variables();
    auto twiddle = saved[0], input = saved[1];
    auto increasing_stride = ctx->saved_data["increasing_stride"].toBool();
    auto result = butterfly_multiply_bw(twiddle, input, grad, increasing_stride);
    auto d_twiddle = std::get<0>(result), d_input = std::get<1>(result);
    return {d_twiddle, d_input, torch::Tensor()};
  }
};

torch::Tensor butterfly_multiply(torch::Tensor twiddle,
                                 torch::Tensor input,
                                 bool increasing_stride) {
  return ButterflyMultiply::apply(twiddle, input, increasing_stride);
}

TORCH_LIBRARY(torch_butterfly, m) {
  m.def("butterfly_multiply_fw", butterfly_multiply_fw);
  m.def("butterfly_multiply_bw", butterfly_multiply_bw);
  m.def("butterfly_multiply", butterfly_multiply);
}
