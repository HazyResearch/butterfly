#include <torch/extension.h>


at::Tensor butterfly_factor_multiply(at::Tensor coefficients, at::Tensor input) {
  // Parameters:
  //     coefficients: (2, 2, n)
  //     input: (batch_size, 2, n)
  auto output = at::empty_like(input);
  auto coefficients_a = coefficients.accessor<float, 3>();
  auto input_a = input.accessor<float, 3>();
  auto output_a = output.accessor<float, 3>();
  for (int b = 0; b < input_a.size(0); ++b) {
    for (int i = 0; i < input_a.size(2); ++i) {
      // float result0 = coefficients_a[0][0][i] * input_a[b][0][i] + coefficients_a[0][1][i] * input_a[b][1][i];
      // float result1 = coefficients_a[1][0][i] * input_a[b][0][i] + coefficients_a[1][1][i] * input_a[b][1][i];
      output_a[b][0][i] = coefficients_a[0][0][i] * input_a[b][0][i] + coefficients_a[0][1][i] * input_a[b][1][i];
      output_a[b][1][i] = coefficients_a[1][0][i] * input_a[b][0][i] + coefficients_a[1][1][i] * input_a[b][1][i];
    }
  }
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("butterfly_factor_multiply", &butterfly_factor_multiply, "Butterfly factor multiply forward");
}
