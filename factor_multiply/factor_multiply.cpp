#include <vector>
#include <torch/extension.h>
// #include <iostream>


at::Tensor butterfly_factor_multiply(at::Tensor coefficients, at::Tensor input) {
  // Parameters:
  //     coefficients: (2, 2, n)
  //     input: (batch_size, 2, n)
  auto output = torch::empty_like(input);
  auto coefficients_a = coefficients.accessor<float, 3>();
  auto input_a = input.accessor<float, 3>();
  auto output_a = output.accessor<float, 3>();
  for (int b = 0; b < input_a.size(0); ++b) {
    for (int i = 0; i < input_a.size(2); ++i) {
      output_a[b][0][i] = coefficients_a[0][0][i] * input_a[b][0][i] + coefficients_a[0][1][i] * input_a[b][1][i];
      output_a[b][1][i] = coefficients_a[1][0][i] * input_a[b][0][i] + coefficients_a[1][1][i] * input_a[b][1][i];
    }
  }
  return output;
}

std::vector<at::Tensor> butterfly_factor_multiply_backward(at::Tensor grad, at::Tensor coefficients, at::Tensor input) {
  // Parameters:
  //     grad: (batch_size, 2, n)
  //     coefficients: (2, 2, n)
  //     input: (batch_size, 2, n)
  auto d_coefficients = torch::zeros_like(coefficients);
  auto d_input = torch::empty_like(input);
  auto grad_a = grad.accessor<float, 3>();
  auto coefficients_a = coefficients.accessor<float, 3>();
  auto input_a = input.accessor<float, 3>();
  auto d_coefficients_a = d_coefficients.accessor<float, 3>();
  auto d_input_a = d_input.accessor<float, 3>();
  for (int b = 0; b < input_a.size(0); ++b) {
    for (int i = 0; i < input_a.size(2); ++i) {
      d_coefficients_a[0][0][i] += grad_a[b][0][i] * input_a[b][0][i];
      d_coefficients_a[0][1][i] += grad_a[b][0][i] * input_a[b][1][i];
      d_coefficients_a[1][0][i] += grad_a[b][1][i] * input_a[b][0][i];
      d_coefficients_a[1][1][i] += grad_a[b][1][i] * input_a[b][1][i];
      d_input_a[b][0][i] = coefficients_a[0][0][i] * grad_a[b][0][i] + coefficients_a[1][0][i] * grad_a[b][1][i];
      d_input_a[b][1][i] = coefficients_a[0][1][i] * grad_a[b][0][i] + coefficients_a[1][1][i] * grad_a[b][1][i];
    }
  }
  return {d_coefficients, d_input};
}

// at::Tensor permutation_factor_even_odd_multiply(at::Scalar p, at::Tensor x, at::Tensor y) {
at::Tensor permutation_factor_even_odd_multiply(float p, at::Tensor input) {
  // Parameters:
  //     p: real number
  //     input: (batch_size, n)
  // Output:
  //     p input + (1 - p) input_permuted
  auto permuted_input = input.reshape({-1, input.size(1) / 2, 2}).transpose(-1, -2);
  input = input.reshape({-1, 2, input.size(1) / 2});
  auto output = torch::empty_like(input);
  auto input_a = input.accessor<float, 3>();
  auto permuted_input_a = permuted_input.accessor<float, 3>();
  auto output_a = output.accessor<float, 3>();
  for (int b = 0; b < input_a.size(0); ++b) {
    for (int i = 0; i < input_a.size(2); ++i) {
      output_a[b][0][i] = (1 - p) * input_a[b][0][i] + p * permuted_input_a[b][0][i];
      output_a[b][1][i] = (1 - p) * input_a[b][1][i] + p * permuted_input_a[b][1][i];
    }
  }
  return output.reshape({-1, 2 * input.size(2)});
}

std::vector<at::Tensor> permutation_factor_even_odd_multiply_backward(at::Tensor grad, float p, at::Tensor input) {
  // Parameters:
  //     grad: (batch_size, n)
  //     p: real number
  //     input: (batch_size, n)
  // Output:
  //     d_p, d_x
  auto batch_size = grad.size(0);
  auto n = grad.size(1);
  auto permuted_input = input.reshape({-1, n / 2, 2}).transpose(-1, -2);
  input = input.reshape({-1, 2, n / 2});
  auto grad_reshaped = grad.reshape({-1, 2, n / 2});
  auto d_p = torch::zeros(1);
  auto permuted_grad = grad.reshape({-1, 2, n / 2}).transpose(-1, -2);
  grad = grad.reshape({-1, n / 2, 2});
  auto d_input = torch::empty_like(grad);
  // Accessors
  auto input_a = input.accessor<float, 3>();
  auto permuted_input_a = permuted_input.accessor<float, 3>();
  auto grad_reshaped_a = grad_reshaped.accessor<float, 3>();
  auto d_p_a = d_p.accessor<float, 1>();
  auto grad_a = grad.accessor<float, 3>();
  auto permuted_grad_a = permuted_grad.accessor<float, 3>();
  auto d_input_a = d_input.accessor<float, 3>();
  for (int b = 0; b < batch_size; ++b) {
    for (int i = 0; i < n / 2; ++i) {
      d_p_a[0] += (permuted_input_a[b][0][i] - input_a[b][0][i]) * grad_reshaped_a[b][0][i]
                  + (permuted_input_a[b][1][i] - input_a[b][1][i]) * grad_reshaped_a[b][1][i];
      d_input_a[b][i][0] = (1 - p) * grad_a[b][i][0] + p * permuted_grad_a[b][i][0];
      d_input_a[b][i][1] = (1 - p) * grad_a[b][i][1] + p * permuted_grad_a[b][i][1];
    }
  }
  return {d_p, d_input.reshape({-1, n})};
}

at::Tensor permutation_factor_reverse_multiply(at::Tensor p, at::Tensor input) {
  // Parameters:
  //     p: (2, )
  //     input: (batch_size, n)
  auto batch_size = input.size(0);
  auto n = input.size(1);
  input = input.reshape({-1, 2, n / 2});
  auto output = torch::empty_like(input);
  auto input_a = input.accessor<float, 3>();
  auto p_a = p.accessor<float, 1>();
  auto output_a = output.accessor<float, 3>();
  for (int b = 0; b < batch_size; ++b) {
    for (int i = 0; i < n / 2; ++i) {
      output_a[b][0][i] = (1 - p_a[0]) * input_a[b][0][i] + p_a[0] * input_a[b][0][n / 2 - 1 - i];
      output_a[b][1][i] = (1 - p_a[1]) * input_a[b][1][i] + p_a[1] * input_a[b][1][n / 2 - 1 - i];
    }
  }
  return output.reshape({-1, n});
}

std::vector<at::Tensor> permutation_factor_reverse_multiply_backward(at::Tensor grad, at::Tensor p, at::Tensor input) {
  // Parameters:
  //     grad: (batch_size, n)
  //     p: (2, )
  //     input: (batch_size, n)
  // Output:
  //     d_p, d_x
  auto batch_size = grad.size(0);
  auto n = grad.size(1);
  input = input.reshape({-1, 2, n / 2});
  grad = grad.reshape({-1, 2, n / 2});
  auto d_p = torch::zeros(2);
  auto d_input = torch::empty_like(input);
  // Accessors
  auto grad_a = grad.accessor<float, 3>();
  auto p_a = p.accessor<float, 1>();
  auto input_a = input.accessor<float, 3>();
  auto d_p_a = d_p.accessor<float, 1>();
  auto d_input_a = d_input.accessor<float, 3>();
  for (int b = 0; b < batch_size; ++b) {
    for (int i = 0; i < n / 2; ++i) {
      d_p_a[0] += (input_a[b][0][n / 2 - 1 - i] - input_a[b][0][i]) * grad_a[b][0][i];
      d_p_a[1] += (input_a[b][1][n / 2 - 1 - i] - input_a[b][1][i]) * grad_a[b][1][i];
      d_input_a[b][0][i] = (1 - p_a[0]) * grad_a[b][0][i] + p_a[0] * grad_a[b][0][n / 2 - 1 - i];
      d_input_a[b][1][i] = (1 - p_a[1]) * grad_a[b][1][i] + p_a[1] * grad_a[b][1][n / 2 - 1 - i];
    }
  }
  return {d_p, d_input.reshape({-1, n})};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("butterfly_factor_multiply", &butterfly_factor_multiply, "Butterfly factor multiply forward");
  m.def("butterfly_factor_multiply_backward", &butterfly_factor_multiply_backward, "Butterfly factor multiply backward");
  m.def("permutation_factor_even_odd_multiply", &permutation_factor_even_odd_multiply, "Permutation factor (even odd) multiply forward");
  m.def("permutation_factor_even_odd_multiply_backward", &permutation_factor_even_odd_multiply_backward, "Permutation factor (even odd) multiply backward");
  m.def("permutation_factor_reverse_multiply", &permutation_factor_reverse_multiply, "Permutation factor (reverse) multiply forward");
  m.def("permutation_factor_reverse_multiply_backward", &permutation_factor_reverse_multiply_backward, "Permutation factor (even odd) multiply backward");
}
