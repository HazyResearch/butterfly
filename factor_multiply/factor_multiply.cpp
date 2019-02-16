#include <immintrin.h>

#include <vector>
#include <torch/extension.h>
// #include <iostream>

void butterfly_factor_multiply_cuda(const at::Tensor& twiddle, const at::Tensor& input, at::Tensor& output);
void butterfly_factor_multiply_backward_cuda(const at::Tensor& grad, const at::Tensor& twiddle, const at::Tensor& input,
                                             at::Tensor& d_twiddle_expanded, at::Tensor& d_input);
void permutation_factor_even_odd_multiply_cuda(const at::Tensor& p, const at::Tensor& input, at::Tensor& output);
void permutation_factor_even_odd_multiply_backward_cuda(const at::Tensor& grad, const at::Tensor& p, const at::Tensor& input,
                                                        at::Tensor& d_p_expanded, at::Tensor& d_input);

at::Tensor butterfly_factor_multiply(const at::Tensor& twiddle, const at::Tensor& input) {
  /* Parameters:
        twiddle: (2, 2, n) if real or (2, 2, n, 2) if complex
        input: (batch_size, 2, n) if real or (batch_size, 2, n, 2) if complex
     Return:
        output: (batch_size, 2, n) if real or (batch_size, 2, n, 2) if complex
  */
  auto output = torch::empty_like(input);
  if (input.is_cuda()) {
    AT_CHECK(twiddle.is_cuda(), "butterfly_factor_multiply: Expected twiddle to be CUDA tensor");
    butterfly_factor_multiply_cuda(twiddle, input, output);
    return output;
  }
  AT_CHECK(!twiddle.is_cuda(), "butterfly_factor_multiply: Expected twiddle to be CPU tensor");
  const auto batch_size = input.size(0);
  const auto n = input.size(2);
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "butterfly_factor_multiply", [&] {
    switch (input.dim()) {
      case 3:  // real
        {
          const auto twiddle_a = twiddle.accessor<scalar_t, 3>();
          const auto input_a = input.accessor<scalar_t, 3>();
          auto output_a = output.accessor<scalar_t, 3>();
          for (int64_t b = 0; b < batch_size; ++b) {
            for (int64_t i = 0; i < n; ++i) {
              const scalar_t twiddle_val[2][2] = {{twiddle_a[0][0][i], twiddle_a[0][1][i]},
                                                  {twiddle_a[1][0][i], twiddle_a[1][1][i]}};
              const scalar_t input_val[2] = {input_a[b][0][i], input_a[b][1][i]};
              for (int64_t j = 0; j <= 1; ++j) {
                output_a[b][j][i] = twiddle_val[j][0] * input_val[0] + twiddle_val[j][1] * input_val[1];
              }
            }
          }
          break;
        }
      case 4:  // complex
        {
          const auto twiddle_a = twiddle.accessor<scalar_t, 4>();
          const auto input_a = input.accessor<scalar_t, 4>();
          auto output_a = output.accessor<scalar_t, 4>();
          for (int64_t b = 0; b < batch_size; ++b) {
            for (int64_t i = 0; i < n; ++i) {
              const scalar_t twiddle_val[2][2][2] = {{{twiddle_a[0][0][i][0], twiddle_a[0][0][i][1]},
                                                      {twiddle_a[0][1][i][0], twiddle_a[0][1][i][1]}},
                                                     {{twiddle_a[1][0][i][0], twiddle_a[1][0][i][1]},
                                                      {twiddle_a[1][1][i][0], twiddle_a[1][1][i][1]}}};
              const scalar_t input_val[2][2] = {{input_a[b][0][i][0], input_a[b][0][i][1]},
                                                {input_a[b][1][i][0], input_a[b][1][i][1]}};
              for (int64_t j = 0; j <= 1; ++j) {
                output_a[b][j][i][0] = twiddle_val[j][0][0] * input_val[0][0] - twiddle_val[j][0][1] * input_val[0][1]
                  + twiddle_val[j][1][0] * input_val[1][0] - twiddle_val[j][1][1] * input_val[1][1];
                output_a[b][j][i][1] = twiddle_val[j][0][0] * input_val[0][1] + twiddle_val[j][0][1] * input_val[0][0]
                  + twiddle_val[j][1][0] * input_val[1][1] + twiddle_val[j][1][1] * input_val[1][0];
              }
            }
          }
          break;
        }
      default:
        AT_ERROR("butterfly_factor_multiply requires input dimension 3 or 4");
    }
  });
  return output;
}

std::vector<at::Tensor> butterfly_factor_multiply_backward(const at::Tensor& grad, const at::Tensor& twiddle, const at::Tensor& input) {
  /* Parameters:
         grad: (batch_size, 2, n) if real or (batch_size, 2, n, 2) if complex
         twiddle: (2, 2, n) if real or (2, 2, n, 2) if complex
         input: (batch_size, 2, n) if real or (batch_size, 2, n, 2) if complex
     Return:
         d_twiddle: (2, 2, n) if real or (2, 2, n, 2) if complex
         d_input: (batch_size, 2, n) if real or (batch_size, 2, n, 2) if complex
  */
  const auto batch_size = input.size(0);
  const auto n = input.size(2);
  auto d_input = torch::empty_like(input);
  if (input.is_cuda()) {
    AT_CHECK(twiddle.is_cuda() && grad.is_cuda(), "butterfly_factor_multiply_backward: Expected grad and twiddle to be CUDA tensor");
    // CUDA kernel will compute the expanded gradient of @twiddle, then we'll call sum over the batch dimension.
    // This is because I haven't figured out how to write efficient reduction kernel in CUDA.
    auto d_twiddle_expanded = input.dim() == 3 ?
      // torch::empty({(batch_size + 3) / 4, 2, 2, n}, torch::dtype(twiddle.dtype()).device(twiddle.device())) :
      torch::empty({batch_size, 2, 2, n}, torch::dtype(twiddle.dtype()).device(twiddle.device())) :
      torch::empty({batch_size, 2, 2, n, 2}, torch::dtype(twiddle.dtype()).device(twiddle.device()));
    butterfly_factor_multiply_backward_cuda(grad, twiddle, input, d_twiddle_expanded, d_input);
    return {d_twiddle_expanded.sum(0), d_input};
  }
  AT_CHECK((!twiddle.is_cuda()) && (!grad.is_cuda()) , "butterfly_factor_multiply_backward: Expected grad and twiddle to be CPU tensor");
  auto d_twiddle = torch::zeros_like(twiddle);
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "butterfly_factor_multiply_backward", [&] {
    switch (input.dim()) {
      case 3:  // real
        {
          const auto grad_a = grad.accessor<scalar_t, 3>();
          const auto twiddle_a = twiddle.accessor<scalar_t, 3>();
          const auto input_a = input.accessor<scalar_t, 3>();
          auto d_twiddle_a = d_twiddle.accessor<scalar_t, 3>();
          auto d_input_a = d_input.accessor<scalar_t, 3>();
          for (int64_t b = 0; b < batch_size; ++b) {
            for (int64_t i = 0; i < n; ++i) {
              const scalar_t twiddle_val[2][2] = {{twiddle_a[0][0][i], twiddle_a[0][1][i]},
                                                  {twiddle_a[1][0][i], twiddle_a[1][1][i]}};
              const scalar_t input_val[2] = {input_a[b][0][i], input_a[b][1][i]};
              const scalar_t grad_val[2] = {grad_a[b][0][i], grad_a[b][1][i]};
              for (int64_t j = 0; j <= 1; ++j) {
                d_twiddle_a[j][0][i] += grad_val[j] * input_val[0];
                d_twiddle_a[j][1][i] += grad_val[j] * input_val[1];
                d_input_a[b][j][i] = twiddle_val[0][j] * grad_val[0] + twiddle_val[1][j] * grad_val[1];
              }
            }
          }
          break;
        }
      case 4:  // complex
        {
          const auto grad_a = grad.accessor<scalar_t, 4>();
          const auto twiddle_a = twiddle.accessor<scalar_t, 4>();
          const auto input_a = input.accessor<scalar_t, 4>();
          auto d_twiddle_a = d_twiddle.accessor<scalar_t, 4>();
          auto d_input_a = d_input.accessor<scalar_t, 4>();
          for (int64_t b = 0; b < batch_size; ++b) {
            for (int64_t i = 0; i < n; ++i) {
              const scalar_t twiddle_val[2][2][2] = {{{twiddle_a[0][0][i][0], twiddle_a[0][0][i][1]},
                                                      {twiddle_a[0][1][i][0], twiddle_a[0][1][i][1]}},
                                                     {{twiddle_a[1][0][i][0], twiddle_a[1][0][i][1]},
                                                      {twiddle_a[1][1][i][0], twiddle_a[1][1][i][1]}}};
              const scalar_t input_val[2][2] = {{input_a[b][0][i][0], input_a[b][0][i][1]},
                                                {input_a[b][1][i][0], input_a[b][1][i][1]}};
              const scalar_t grad_val[2][2] = {{grad_a[b][0][i][0], grad_a[b][0][i][1]},
                                               {grad_a[b][1][i][0], grad_a[b][1][i][1]}};
              for (int64_t j = 0; j <= 1; ++j) {
                // Multiply by complex conjugate
                d_twiddle_a[j][0][i][0] += grad_val[j][0] * input_val[0][0] + grad_val[j][1] * input_val[0][1];
                d_twiddle_a[j][0][i][1] += -grad_val[j][0] * input_val[0][1] + grad_val[j][1] * input_val[0][0];
                d_twiddle_a[j][1][i][0] += grad_val[j][0] * input_val[1][0] + grad_val[j][1] * input_val[1][1];
                d_twiddle_a[j][1][i][1] += -grad_val[j][0] * input_val[1][1] + grad_val[j][1] * input_val[1][0];
                d_input_a[b][j][i][0] = twiddle_val[0][j][0] * grad_val[0][0] + twiddle_val[0][j][1] * grad_val[0][1]
                  + twiddle_val[1][j][0] * grad_val[1][0] + twiddle_val[1][j][1] * grad_val[1][1];
                d_input_a[b][j][i][1] = twiddle_val[0][j][0] * grad_val[0][1] - twiddle_val[0][j][1] * grad_val[0][0]
                  + twiddle_val[1][j][0] * grad_val[1][1] - twiddle_val[1][j][1] * grad_val[1][0];
              }
            }
          }
          break;
        }
      default:
        AT_ERROR("butterfly_factor_multiply_backward requires input dimension 3 or 4");
    }
  });
  return {d_twiddle, d_input};
}

at::Tensor permutation_factor_even_odd_multiply(const at::Tensor& p, const at::Tensor& input) {
  /* Parameters:
         p: (1, )
         input: (batch_size, n) if real or (batch_size, n, 2) if complex
     Output:
         p input + (1 - p) input_permuted: (batch_size, n) if real or (batch_size, n, 2) if complex
  */
  auto output = torch::empty_like(input);
  if (input.is_cuda()) {
    permutation_factor_even_odd_multiply_cuda(p, input, output);
    return output;
  }
  const auto batch_size = input.size(0);
  const auto n = input.size(1);
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "permutation_factor_even_odd_multiply", [&] {
    const scalar_t p_a = p.accessor<scalar_t, 1>()[0];
    switch (input.dim()) {
      case 2: // real
        {
          const auto permuted_input = input.reshape({batch_size, n / 2, 2}).transpose(1, 2);
          const auto input_folded = input.reshape({batch_size, 2, n / 2});
          output = output.view({batch_size, 2, n / 2});
          const auto input_a = input_folded.accessor<scalar_t, 3>();
          const auto permuted_input_a = permuted_input.accessor<scalar_t, 3>();
          auto output_a = output.accessor<scalar_t, 3>();
          for (int64_t b = 0; b < batch_size; ++b) {
            for (int64_t i = 0; i < n / 2; ++i) {
              // Manually unrolling loop seems to be faster
              output_a[b][0][i] = (1 - p_a) * input_a[b][0][i] + p_a * permuted_input_a[b][0][i];
              output_a[b][1][i] = (1 - p_a) * input_a[b][1][i] + p_a * permuted_input_a[b][1][i];
            }
          }
          output = output.view({batch_size, n});
          break;
        }
      case 3: // complex
        {
          const auto permuted_input = input.reshape({batch_size, n / 2, 2, 2}).transpose(1, 2);
          const auto input_folded = input.reshape({batch_size, 2, n / 2, 2});
          output = output.view({batch_size, 2, n / 2, 2});
          const auto input_a = input_folded.accessor<scalar_t, 4>();
          const auto permuted_input_a = permuted_input.accessor<scalar_t, 4>();
          auto output_a = output.accessor<scalar_t, 4>();
          for (int64_t b = 0; b < batch_size; ++b) {
            for (int64_t i = 0; i < n / 2; ++i) {
              output_a[b][0][i][0] = (1 - p_a) * input_a[b][0][i][0] + p_a * permuted_input_a[b][0][i][0];
              output_a[b][0][i][1] = (1 - p_a) * input_a[b][0][i][1] + p_a * permuted_input_a[b][0][i][1];
              output_a[b][1][i][0] = (1 - p_a) * input_a[b][1][i][0] + p_a * permuted_input_a[b][1][i][0];
              output_a[b][1][i][1] = (1 - p_a) * input_a[b][1][i][1] + p_a * permuted_input_a[b][1][i][1];
            }
          }
          output = output.view({batch_size, n, 2});
          break;
        }
      default:
        AT_ERROR("permutation_factor_even_odd_multiply requires input dimension 2 or 3");
    }
  });
  return output;
}

std::vector<at::Tensor> permutation_factor_even_odd_multiply_backward(const at::Tensor& grad, const at::Tensor& p, const at::Tensor& input) {
  /* Parameters:
         grad: (batch_size, n) if real or (batch_size, n, 2) if complex
         p: (1, )
         input: (batch_size, n) if real or (batch_size, n, 2) if complex
     Output:
         d_p: (1, )
         d_input: (batch_size, n) if real or (batch_size, n, 2) if complex
  */
  const auto batch_size = grad.size(0);
  const auto n = grad.size(1);
  auto d_input = torch::empty_like(input);
  auto d_p = torch::zeros_like(p);
  if (input.is_cuda()) {
    AT_CHECK(grad.is_cuda(), "permutation_factor_even_odd_multiply_backward: Expected grad to be CUDA tensor");
    // CUDA kernel will compute the expanded gradient of @p, then we'll call sum.
    // This is because I haven't figured out how to write efficient reduction kernel in CUDA.
    auto d_p_expanded = torch::empty({batch_size, n / 2}, torch::dtype(input.dtype()).device(input.device()));
    permutation_factor_even_odd_multiply_backward_cuda(grad, p, input, d_p_expanded, d_input);
    d_p[0] = d_p_expanded.sum();
    return {d_p, d_input};
  }
  AT_CHECK(!grad.is_cuda(), "butterfly_factor_multiply: Expected grad to be CPU tensor");
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "permutation_factor_even_odd_multiply", [&] {
    const scalar_t p_a = p.accessor<scalar_t, 1>()[0];
    auto d_p_a = d_p.accessor<scalar_t, 1>();
    scalar_t d_p_temp = 0;
    switch (input.dim()) {
      case 2: // real
        {
          const auto permuted_input = input.reshape({batch_size, n / 2, 2}).transpose(1, 2);
          const auto input_folded = input.reshape({batch_size, 2, n / 2});
          const auto grad_reshaped = grad.reshape({batch_size, 2, n / 2});
          const auto permuted_grad = grad.reshape({batch_size, 2, n / 2}).transpose(1, 2);
          const auto grad_folded = grad.reshape({batch_size, n / 2, 2});
          d_input = d_input.view({batch_size, n/ 2, 2});
          // Accessors
          const auto input_a = input_folded.accessor<scalar_t, 3>();
          const auto permuted_input_a = permuted_input.accessor<scalar_t, 3>();
          const auto grad_reshaped_a = grad_reshaped.accessor<scalar_t, 3>();
          const auto grad_a = grad_folded.accessor<scalar_t, 3>();
          const auto permuted_grad_a = permuted_grad.accessor<scalar_t, 3>();
          auto d_input_a = d_input.accessor<scalar_t, 3>();
          for (int64_t b = 0; b < batch_size; ++b) {
            for (int64_t i = 0; i < n / 2; ++i) {
              d_p_temp += (permuted_input_a[b][0][i] - input_a[b][0][i]) * grad_reshaped_a[b][0][i]
                + (permuted_input_a[b][1][i] - input_a[b][1][i]) * grad_reshaped_a[b][1][i];
              d_input_a[b][i][0] = (1 - p_a) * grad_a[b][i][0] + p_a * permuted_grad_a[b][i][0];
              d_input_a[b][i][1] = (1 - p_a) * grad_a[b][i][1] + p_a * permuted_grad_a[b][i][1];
            }
          }
          d_input = d_input.view({batch_size, n});
          break;
        }
      case 3: // complex
        {
          const auto permuted_input = input.reshape({batch_size, n / 2, 2, 2}).transpose(1, 2);
          const auto input_folded = input.reshape({batch_size, 2, n / 2, 2});
          const auto grad_reshaped = grad.reshape({batch_size, 2, n / 2, 2});
          const auto permuted_grad = grad.reshape({batch_size, 2, n / 2, 2}).transpose(1, 2);
          const auto grad_folded = grad.reshape({batch_size, n / 2, 2, 2});
          d_input = d_input.view({batch_size, n/ 2, 2, 2});
          // Accessors
          const auto input_a = input_folded.accessor<scalar_t, 4>();
          const auto permuted_input_a = permuted_input.accessor<scalar_t, 4>();
          const auto grad_reshaped_a = grad_reshaped.accessor<scalar_t, 4>();
          const auto grad_a = grad_folded.accessor<scalar_t, 4>();
          const auto permuted_grad_a = permuted_grad.accessor<scalar_t, 4>();
          auto d_input_a = d_input.accessor<scalar_t, 4>();
          for (int64_t b = 0; b < batch_size; ++b) {
            for (int64_t i = 0; i < n / 2; ++i) {
              d_p_temp += (permuted_input_a[b][0][i][0] - input_a[b][0][i][0]) * grad_reshaped_a[b][0][i][0]
                + (permuted_input_a[b][0][i][1] - input_a[b][0][i][1]) * grad_reshaped_a[b][0][i][1]
                + (permuted_input_a[b][1][i][0] - input_a[b][1][i][0]) * grad_reshaped_a[b][1][i][0]
                + (permuted_input_a[b][1][i][1] - input_a[b][1][i][1]) * grad_reshaped_a[b][1][i][1];
              d_input_a[b][i][0][0] = (1 - p_a) * grad_a[b][i][0][0] + p_a * permuted_grad_a[b][i][0][0];
              d_input_a[b][i][0][1] = (1 - p_a) * grad_a[b][i][0][1] + p_a * permuted_grad_a[b][i][0][1];
              d_input_a[b][i][1][0] = (1 - p_a) * grad_a[b][i][1][0] + p_a * permuted_grad_a[b][i][1][0];
              d_input_a[b][i][1][1] = (1 - p_a) * grad_a[b][i][1][1] + p_a * permuted_grad_a[b][i][1][1];
            }
          }
          d_input = d_input.view({batch_size, n, 2});
          break;
        }
      default:
        AT_ERROR("permutation_factor_even_odd_multiply_backward requires input dimension 2 or 3");
    }
    d_p_a[0] = d_p_temp;
  });
  return {d_p, d_input};
}

at::Tensor permutation_factor_reverse_multiply(const at::Tensor& p, const at::Tensor& input) {
  /* Parameters:
         p: (2, )
         input: (batch_size, n) if real or (batch_size, n, 2) if complex
     Output:
         p input + (1 - p) input_reversed: (batch_size, n) if real or (batch_size, n, 2) if complex
  */
  auto output = torch::empty_like(input);
  // if (input.is_cuda()) {
  //   permutation_factor_even_odd_multiply_cuda(p, input, output);
  //   return output;
  // }
  const auto batch_size = input.size(0);
  const auto n = input.size(1);
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "permutation_factor_reverse_multiply", [&] {
    const scalar_t p_a[2] = {p.accessor<float, 1>()[0], p.accessor<float, 1>()[1]};
    switch (input.dim()) {
      case 2: // real
        {
          const auto input_folded = input.reshape({batch_size, 2, n / 2});
          output = output.view({batch_size, 2, n / 2});
          const auto input_a = input_folded.accessor<scalar_t, 3>();
          auto output_a = output.accessor<scalar_t, 3>();
          for (int64_t b = 0; b < batch_size; ++b) {
            for (int64_t i = 0; i < n / 4; ++i) {
              // output_a[b][0][i] = (1 - p_a[0]) * input_a[b][0][i] + p_a[0] * input_a[b][0][n / 2 - 1 - i];
              const scalar_t in0[2] = {input_a[b][0][i], input_a[b][0][n / 2 - 1 - i]};
              output_a[b][0][i] = (1 - p_a[0]) * in0[0] + p_a[0] * in0[1];
              output_a[b][0][n / 2 - 1 - i] = p_a[0] * in0[0] + (1 - p_a[0]) * in0[1];
              // output_a[b][1][i] = (1 - p_a[1]) * input_a[b][1][i] + p_a[1] * input_a[b][1][n / 2 - 1 - i];
              const scalar_t in1[2] = {input_a[b][1][i], input_a[b][1][n / 2 - 1 - i]};
              output_a[b][1][i] = (1 - p_a[1]) * in1[0] + p_a[1] * in1[1];
              output_a[b][1][n / 2 - 1 - i] = p_a[1] * in1[0] + (1 - p_a[1]) * in1[1];
            }
          }
          output = output.view({batch_size, n});
          break;
        }
      case 3: // complex
        {
          const auto input_folded = input.reshape({batch_size, 2, n / 2, 2});
          output = output.view({batch_size, 2, n / 2, 2});
          const auto input_a = input_folded.accessor<scalar_t, 4>();
          auto output_a = output.accessor<scalar_t, 4>();
          for (int64_t b = 0; b < batch_size; ++b) {
            for (int64_t i = 0; i < n / 4; ++i) {
              const scalar_t in00[2] = {input_a[b][0][i][0], input_a[b][0][n / 2 - 1 - i][0]};
              output_a[b][0][i][0] = (1 - p_a[0]) * in00[0] + p_a[0] * in00[1];
              output_a[b][0][n / 2 - 1 - i][0] = p_a[0] * in00[0] + (1 - p_a[0]) * in00[1];
              const scalar_t in01[2] = {input_a[b][0][i][1], input_a[b][0][n / 2 - 1 - i][1]};
              output_a[b][0][i][1] = (1 - p_a[0]) * in01[0] + p_a[0] * in01[1];
              output_a[b][0][n / 2 - 1 - i][1] = p_a[0] * in01[0] + (1 - p_a[0]) * in01[1];
              const scalar_t in10[2] = {input_a[b][1][i][0], input_a[b][1][n / 2 - 1 - i][0]};
              output_a[b][1][i][0] = (1 - p_a[1]) * in10[0] + p_a[1] * in10[1];
              output_a[b][1][n / 2 - 1 - i][0] = p_a[1] * in10[0] + (1 - p_a[1]) * in10[1];
              const scalar_t in11[2] = {input_a[b][1][i][1], input_a[b][1][n / 2 - 1 - i][1]};
              output_a[b][1][i][1] = (1 - p_a[1]) * in11[0] + p_a[1] * in11[1];
              output_a[b][1][n / 2 - 1 - i][1] = p_a[1] * in11[0] + (1 - p_a[1]) * in11[1];
            }
          }
          output = output.view({batch_size, n, 2});
          break;
        }
      default:
        AT_ERROR("permutation_factor_reverse_multiply requires input dimension 2 or 3");
    }
  });
  return output;
}

std::vector<at::Tensor> permutation_factor_reverse_multiply_backward(const at::Tensor& grad, const at::Tensor& p, const at::Tensor& input) {
  /* Parameters:
        grad: (batch_size, n) if real or (batch_size, n, 2) if complex
        p: (2, )
        input: (batch_size, n) if real or (batch_size, n, 2) if complex
     Output:
        d_p: (2, )
        d_input: (batch_size, n) if real or (batch_size, n, 2) if complex
  */
  const auto batch_size = grad.size(0);
  const auto n = grad.size(1);
  auto d_input = torch::empty_like(input);
  auto d_p = torch::zeros_like(p);
  // if (input.is_cuda()) {
  //   AT_CHECK(grad.is_cuda(), "permutation_factor_reverse_multiply_backward: Expected grad to be CUDA tensor");
  //   // CUDA kernel will compute the expanded gradient of @p, then we'll call sum.
  //   // This is because I haven't figured out how to write efficient reduction kernel in CUDA.
  //   auto d_p_expanded = torch::empty({batch_size, n / 2}, torch::dtype(input.dtype()).device(input.device()));
  //   permutation_factor_even_odd_multiply_backward_cuda(grad, p, input, d_p_expanded, d_input);
  //   d_p[0] = d_p_expanded.sum();
  //   return {d_p, d_input};
  // }
  // AT_CHECK(!grad.is_cuda(), "butterfly_factor_multiply: Expected grad to be CPU tensor");
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "permutation_factor_even_odd_multiply_backward", [&] {
    const scalar_t p_a[2] = {p.accessor<scalar_t, 1>()[0], p.accessor<scalar_t, 1>()[1]};
    auto d_p_a = d_p.accessor<scalar_t, 1>();
    scalar_t d_p_temp[2] = {0, 0};
    switch (input.dim()) {
      case 2: // real
        {
          const auto input_folded = input.reshape({batch_size, 2, n / 2});
          const auto grad_folded = grad.reshape({batch_size, 2, n / 2});
          d_input = d_input.view({batch_size, 2, n / 2});
          // Accessors
          const auto input_a = input_folded.accessor<scalar_t, 3>();
          const auto grad_a = grad_folded.accessor<scalar_t, 3>();
          auto d_input_a = d_input.accessor<scalar_t, 3>();
          for (int64_t b = 0; b < batch_size; ++b) {
            for (int64_t i = 0; i < n / 4; ++i) {
              const scalar_t in0[2] = {input_a[b][0][i], input_a[b][0][n / 2 - 1 - i]};
              const scalar_t g0[2] = {grad_a[b][0][i], grad_a[b][0][n / 2 - 1 - i]};
              d_p_temp[0] += (in0[1] - in0[0]) * (g0[0] - g0[1]);
              d_input_a[b][0][i] = (1 - p_a[0]) * g0[0] + p_a[0] * g0[1];
              d_input_a[b][0][n / 2 - 1 - i] = p_a[0] * g0[0] + (1 - p_a[0]) * g0[1];
              const scalar_t in1[2] = {input_a[b][1][i], input_a[b][1][n / 2 - 1 - i]};
              const scalar_t g1[2] = {grad_a[b][1][i], grad_a[b][1][n / 2 - 1 - i]};
              d_p_temp[1] += (in1[1] - in1[0]) * (g1[0] - g1[1]);
              d_input_a[b][1][i] = (1 - p_a[1]) * g1[0] + p_a[1] * g1[1];
              d_input_a[b][1][n / 2 - 1 - i] = p_a[1] * g1[0] + (1 - p_a[1]) * g1[1];
            }
          }
          d_input = d_input.view({batch_size, n});
          break;
        }
      case 3: // complex
        {
          const auto input_folded = input.reshape({batch_size, 2, n / 2, 2});
          const auto grad_folded = grad.reshape({batch_size, n / 2, 2, 2});
          d_input = d_input.view({batch_size, n/ 2, 2, 2});
          // Accessors
          const auto input_a = input_folded.accessor<scalar_t, 4>();
          const auto grad_a = grad_folded.accessor<scalar_t, 4>();
          auto d_input_a = d_input.accessor<scalar_t, 4>();
          for (int64_t b = 0; b < batch_size; ++b) {
            for (int64_t i = 0; i < n / 4; ++i) {
              const scalar_t in00[2] = {input_a[b][0][i][0], input_a[b][0][n / 2 - 1 - i][0]};
              const scalar_t g00[2] = {grad_a[b][0][i][0], grad_a[b][0][n / 2 - 1 - i][0]};
              d_p_temp[0] += (in00[1] - in00[0]) * (g00[0] - g00[1]);
              d_input_a[b][0][i][0] = (1 - p_a[0]) * g00[0] + p_a[0] * g00[1];
              d_input_a[b][0][n / 2 - 1 - i][0] = p_a[0] * g00[0] + (1 - p_a[0]) * g00[1];
              const scalar_t in01[2] = {input_a[b][0][i][1], input_a[b][0][n / 2 - 1 - i][1]};
              const scalar_t g01[2] = {grad_a[b][0][i][1], grad_a[b][0][n / 2 - 1 - i][1]};
              d_p_temp[0] += (in01[1] - in01[0]) * (g01[0] - g01[1]);
              d_input_a[b][0][i][1] = (1 - p_a[0]) * g01[0] + p_a[0] * g01[1];
              d_input_a[b][0][n / 2 - 1 - i][1] = p_a[0] * g01[0] + (1 - p_a[0]) * g01[1];
              const scalar_t in10[2] = {input_a[b][1][i][0], input_a[b][1][n / 2 - 1 - i][0]};
              const scalar_t g10[2] = {grad_a[b][1][i][0], grad_a[b][1][n / 2 - 1 - i][0]};
              d_p_temp[1] += (in10[1] - in10[0]) * (g10[0] - g10[1]);
              d_input_a[b][1][i][0] = (1 - p_a[1]) * g10[0] + p_a[1] * g10[1];
              d_input_a[b][1][n / 2 - 1 - i][0] = p_a[1] * g10[0] + (1 - p_a[1]) * g10[1];
              const scalar_t in11[2] = {input_a[b][1][i][1], input_a[b][1][n / 2 - 1 - i][1]};
              const scalar_t g11[2] = {grad_a[b][1][i][1], grad_a[b][1][n / 2 - 1 - i][1]};
              d_p_temp[1] += (in11[1] - in11[0]) * (g11[0] - g11[1]);
              d_input_a[b][1][i][1] = (1 - p_a[1]) * g11[0] + p_a[1] * g11[1];
              d_input_a[b][1][n / 2 - 1 - i][1] = p_a[1] * g11[0] + (1 - p_a[1]) * g11[1];
            }
          }
          d_input = d_input.view({batch_size, n, 2});
          break;
        }
      default:
        AT_ERROR("permutation_factor_reverse_multiply_backward requires input dimension 2 or 3");
    }
    d_p_a[0] = d_p_temp[0];
    d_p_a[1] = d_p_temp[1];
  });
  return {d_p, d_input};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("butterfly_factor_multiply", &butterfly_factor_multiply, "Butterfly factor multiply forward");
  m.def("butterfly_factor_multiply_backward", &butterfly_factor_multiply_backward, "Butterfly factor multiply backward");
  m.def("permutation_factor_even_odd_multiply", &permutation_factor_even_odd_multiply, "Permutation factor (even odd) multiply forward");
  m.def("permutation_factor_even_odd_multiply_backward", &permutation_factor_even_odd_multiply_backward, "Permutation factor (even odd) multiply backward");
  m.def("permutation_factor_reverse_multiply", &permutation_factor_reverse_multiply, "Permutation factor (reverse) multiply forward");
  m.def("permutation_factor_reverse_multiply_backward", &permutation_factor_reverse_multiply_backward, "Permutation factor (even odd) multiply backward");
}

// Gives segfault right now
at::Tensor butterfly_factor_multiply_256(const at::Tensor& twiddle, const at::Tensor& input) {
  /* Parameters:
        twiddle: (2, 2, n)
        input: (batch_size, 2, n)
     Return:
        output: (batch_size, 2, n)
  */
  auto batch_size = input.size(0);
  auto n = input.size(2);
  auto output = torch::empty_like(input);
  if (n % 8 != 0) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "butterfly_factor_multiply", [&] {
    auto twiddle_a = twiddle.accessor<scalar_t, 3>();
    auto input_a = input.accessor<scalar_t, 3>();
    auto output_a = output.accessor<scalar_t, 3>();
    for (int64_t b = 0; b < batch_size; ++b) {
      for (int64_t i = 0; i < n; ++i) {
        output_a[b][0][i] = twiddle_a[0][0][i] * input_a[b][0][i] + twiddle_a[0][1][i] * input_a[b][1][i];
        output_a[b][1][i] = twiddle_a[1][0][i] * input_a[b][0][i] + twiddle_a[1][1][i] * input_a[b][1][i];
      }
    }
  });
  } else {
    float* coefficients_data = twiddle.data<float>();
    float* input_data = input.data<float>();
    float* output_data = output.data<float>();
    auto coefficients_stride_0 = twiddle.stride(0);
    auto coefficients_stride_1 = twiddle.stride(1);
    auto coefficients_stride_2 = twiddle.stride(2);
    auto input_stride_0 = input.stride(0);
    auto input_stride_1 = input.stride(1);
    auto input_stride_2 = input.stride(2);
    auto output_stride_0 = output.stride(0);
    auto output_stride_1 = output.stride(1);
    auto output_stride_2 = output.stride(2);
    for (int64_t b = 0; b < batch_size; ++b) {
      for (int64_t i = 0; i < n; i += 8) {
        __m256 coef00 = _mm256_load_ps(coefficients_data + i * coefficients_stride_2);
        __m256 coef01 = _mm256_load_ps(coefficients_data + coefficients_stride_1 + i * coefficients_stride_2);
        __m256 coef10 = _mm256_load_ps(coefficients_data + coefficients_stride_0 + i * coefficients_stride_2);
        __m256 coef11 = _mm256_load_ps(coefficients_data + coefficients_stride_0 + coefficients_stride_1 + i * coefficients_stride_2);
        __m256 input0 = _mm256_load_ps(input_data + b * input_stride_0 + i * input_stride_2);
        __m256 input1 = _mm256_load_ps(input_data + b * input_stride_0 + input_stride_1 + i * input_stride_2);
        __m256 output0 = _mm256_add_ps(_mm256_mul_ps(coef00, input0), _mm256_mul_ps(coef01, input1));
        __m256 output1 = _mm256_add_ps(_mm256_mul_ps(coef10, input0), _mm256_mul_ps(coef11, input1));
        _mm256_store_ps(output_data + b * output_stride_0 + i * output_stride_2, output0);
        _mm256_store_ps(output_data + b * output_stride_0 + output_stride_1 + i * output_stride_2, output1);
      }
    }
  }
  return output;
}
