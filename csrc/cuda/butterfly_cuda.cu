#include "butterfly_cuda.h"
#include "complex_utils.cuh"
#include <ATen/cuda/CUDAContext.h>  // For getCurrentCUDAStream
#include <THC/THCAtomics.cuh>  // For atomicAdd on complex
#include "map.h"

// Only support float (not double) for now to speed up compilation time
// https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/Dispatch.h
#undef AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES
#define AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(TYPE, NAME, ...)                                  \
  [&] {                                                                                          \
    const auto& the_type = TYPE;                                                                 \
    /* don't use TYPE again in case it is an expensive or side-effect op */                      \
    at::ScalarType _st = ::detail::scalar_type(the_type);                                        \
    RECORD_KERNEL_FUNCTION_DTYPE(NAME, _st);                                                     \
    switch (_st) {                                                                               \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Float, float, __VA_ARGS__)                      \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::ComplexFloat, c10::complex<float>, __VA_ARGS__) \
      default:                                                                                   \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");                           \
    }                                                                                            \
  }()


#define FULL_MASK 0xffffffff

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
  static constexpr int SMEM_PER_MP = 96 * (1 << 10);
#else
  static constexpr int SMEM_PER_MP = 64 * (1 << 10);
#endif
static constexpr int WARP_SIZE = 32;
// static constexpr int SMEM_PER_MP = 64 * (1 << 10);
static constexpr int MAX_SMEM_PER_BLOCK = 48 * (1 << 10);
static constexpr int MAX_BLOCK_SIZE = 1024;
// static constexpr int WORK_PER_THREAD = 16;
// static constexpr int ELEMENTARY_SIZE = MAX_BLOCK_SIZE / 2;
// static constexpr int MAX_N_FACTORS = 10;
static constexpr int MAX5_FORWARD_BLOCK_SIZE = 512;
static constexpr int MAX5_BACKWARD_BLOCK_SIZE = 256;
static constexpr int ITEMS_PER_THREAD_FORWARD[14] = {4, 4, 4, 4, 4, 4, 4, 8, 8, 8, 13, 10, 4, 4};
static constexpr int ITEMS_PER_THREAD_BACKWARD[14] = {16, 16, 16, 16, 16, 16, 16, 16, 16, 8, 8, 8, 8, 8};
static constexpr int ITEMS_PER_THREAD_FORWARD_MAX5[9] = {1, 2, 2, 4, 4, 4, 2, 1, 1};
static constexpr int ITEMS_PER_THREAD_BACKWARD_MAX5[8] = {8, 8, 8, 8, 8, 6, 8, 3};
static constexpr int MIN_BLOCKS_PER_MP_FORWARD[14] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1};
static constexpr int MIN_BLOCKS_PER_MP_BACKWARD[14] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
  static constexpr int ITEMS_PER_THREAD_ORTHO_FORWARD[14] = {4, 4, 4, 4, 4, 4, 4, 8, 16, 16, 16, 10, 4, 4};
  static constexpr int ITEMS_PER_THREAD_ORTHO_BACKWARD[14] = {16, 16, 16, 16, 16, 16, 16, 16, 8, 16, 8, 8, 8, 8};
  static constexpr int MIN_BLOCKS_PER_MP_ORTHO_FORWARD[14] = {1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1};
  static constexpr int MIN_BLOCKS_PER_MP_ORTHO_BACKWARD[14] = {1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1};
#else
  static constexpr int ITEMS_PER_THREAD_ORTHO_FORWARD[14] = {4, 4, 4, 4, 4, 4, 4, 8, 8, 8, 8, 10, 4, 4};
  static constexpr int ITEMS_PER_THREAD_ORTHO_BACKWARD[14] = {16, 16, 16, 16, 16, 16, 16, 16, 8, 8, 8, 8, 8, 8};
  static constexpr int MIN_BLOCKS_PER_MP_ORTHO_FORWARD[14] = {1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1};
  static constexpr int MIN_BLOCKS_PER_MP_ORTHO_BACKWARD[14] = {1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1};
#endif

// Need to have general template, but construction will yield compilation error.
template <typename scalar_t> struct maxstep {maxstep() = delete;};
template <> struct maxstep<float> {
  static constexpr int MAXSTEP_FW = 9;
  static constexpr int MAXSTEP_BW = 8;
};
template <> struct maxstep<double> {
  static constexpr int MAXSTEP_FW = 8;
  static constexpr int MAXSTEP_BW = 7;
};
template <> struct maxstep<c10::complex<float>> {
  static constexpr int MAXSTEP_FW = 8;
  static constexpr int MAXSTEP_BW = 7;
};

// This takes a lambda templated on int, and a number n between 1 and n_max, and call that lambda templated on n.
// The lambda is templated using generic lambda (C++14), using std::integral_constant to encode the int template as type.
// What Dispatch<n_max, decltype(lambda_int)>::call(lambda_int, n) does is equivalent to this:
// switch (n):
//   {
//     case 1: lamda_int(std::integral_type<int, 1>()); break;
//     case 2: lamda_int(std::integral_type<int, 2>()); break;
//     ...
//     case n_max: lamda_int(std::integral_type<int, n_max>()); break;
//   }
// The difficulty is of course how to do this for a constexpr n_max.
// I would have preferred to write this as function but partial template speciallization isn't support for functions.
template<int n_max, typename Function>
struct Dispatch {
  static void inline call(Function lambda_int, int n) {
    n == n_max ? lambda_int(std::integral_constant<int, n_max>()) : Dispatch<n_max - 1, Function>::call(lambda_int, n);
  }
};
template<typename Function>
struct Dispatch<0, Function> {
  static void inline call(Function lambda_int, int n) {}
};

template <typename T, size_t N>
using CudaAcsr = at::GenericPackedTensorAccessor<T, N, at::RestrictPtrTraits, int32_t>;

constexpr __host__ __device__ int min_const(int x, int y) { return x <= y ? x : y; }
constexpr __host__ __device__ int min_const(int x, int y, int z) { return min_const(min_const(x, y), z); }
constexpr __host__ __device__ int max_const(int x, int y) { return x >= y ? x : y; }
constexpr __host__ __device__ int div_up_const(int a, int b) { return (a + b - 1) / b; }

// __host__ __device__ static inline int64_t div_up(int64_t a, int64_t b) {
//   return (a + b - 1) / b;
// }

__host__ __device__ static inline int div_up(int a, int b) {
  return (a + b - 1) / b;
}

// Delete the bit at position @position in the binary representation of x
__host__ __device__ static inline int delete_bit(int x, unsigned char position) {
  int mask = (1 << position) - 1;
  return ((x >> 1) & ~mask) | (x & mask);
  // return ((x >> (position + 1)) << position) + (x & ((1 << position) - 1));
}

template<typename scalar_t>
__device__ __forceinline__ void initalize_shared_mem(scalar_t mem[], int size) {
  // Assume that block only uses x and y coordinates, not z coordinate
  for (int t = threadIdx.x + threadIdx.y * blockDim.x; t < size; t += blockDim.x * blockDim.y) {
    mem[t] = 0;
  }
}

template<typename scalar_t> struct InputReader {
  const CudaAcsr<scalar_t, 3> input_a;
  const int batch_size, input_size;

  InputReader(const torch::Tensor input):
    input_a(input.packed_accessor32<scalar_t, 3, at::RestrictPtrTraits>()),
      batch_size(input.size(0)), input_size(input.size(2)) {}

  InputReader(const CudaAcsr<scalar_t, 3> &input_a):
    input_a(input_a), batch_size(input_a.size(0)), input_size(input_a.size(2)) {}

  template<int items_per_thread, int mult_per_warp=1>
  __device__ __forceinline__ void load_max5(scalar_t input_val[mult_per_warp][items_per_thread],
                                            int batch_idx_start, int input_idx_start, int input_idx_stride) {
    const int s = blockIdx.z;
    #pragma unroll
    for (int mult = 0; mult < mult_per_warp; mult++) {
      int i = mult * input_idx_stride + input_idx_start;
      #pragma unroll
      for (int item = 0; item < items_per_thread; item++){
        input_val[mult][item] = (batch_idx_start + item < batch_size && i < input_size) ?
          input_a[batch_idx_start + item][s][i] : 0;
      }
    }
  }

};

template<typename scalar_t> struct OutputWriter {
  CudaAcsr<scalar_t, 3> output_a;
  const int batch_size, output_size;

  OutputWriter(torch::Tensor output):
    output_a(output.packed_accessor32<scalar_t, 3, at::RestrictPtrTraits>()),
      batch_size(output.size(0)), output_size(output.size(2)) {}

  OutputWriter(CudaAcsr<scalar_t, 3> &output_a):
    output_a(output_a), batch_size(output_a.size(0)), output_size(output_a.size(2)) {}

  template<int items_per_thread, int mult_per_warp=1>
  __device__ __forceinline__ void save_max5(scalar_t output_val[mult_per_warp][items_per_thread],
                                            int batch_idx_start, int input_idx_start, int input_idx_stride) {
    const int s = blockIdx.z;
    #pragma unroll
    for (int mult = 0; mult < mult_per_warp; mult++) {
      int i = mult * input_idx_stride + input_idx_start;
      if (i < output_size) {
        #pragma unroll
        for (int item = 0; (item < items_per_thread) && (batch_idx_start + item < batch_size); item++){
          output_a[batch_idx_start + item][s][i] = output_val[mult][item];
        }
      }
    }
  }

};

template<typename scalar_t> struct TwiddleReader {
  const CudaAcsr<scalar_t, 5> twiddle_a;

  TwiddleReader(const torch::Tensor twiddle):
    twiddle_a(twiddle.packed_accessor32<scalar_t, 5, at::RestrictPtrTraits>()) {}

  template<int nsteps, bool increasing_stride>
  __device__ __forceinline__ void load_max5(scalar_t s_twiddle[nsteps][2][1 << nsteps],
                                            int input_idx_start_bit, int low_bits, int high_bits) {
    constexpr int span = 1 << nsteps;
    const int s = blockIdx.y + gridDim.y * blockIdx.z;  // For conv2d butterfly as well
    for (int t = threadIdx.x + threadIdx.y * blockDim.x; t < nsteps * (span / 2); t += blockDim.x * blockDim.y) {
      const int step = t / (span / 2);
      const int s_twiddle_stride = 1 << (increasing_stride ? step : nsteps - 1 - step);
      const int remainder = t % (span / 2);
      const int low_order_bits = remainder & (s_twiddle_stride - 1);
      const int s_idx = 2 * (remainder - low_order_bits) + low_order_bits;
      const int idx = (high_bits >> 1) | (remainder << input_idx_start_bit) | low_bits;
      s_twiddle[step][0][s_idx] = twiddle_a[s][step][idx][0][0];
      s_twiddle[step][1][s_idx] = twiddle_a[s][step][idx][0][1];
      s_twiddle[step][1][s_idx + s_twiddle_stride] = twiddle_a[s][step][idx][1][0];
      s_twiddle[step][0][s_idx + s_twiddle_stride] = twiddle_a[s][step][idx][1][1];
    }
  }

};

template<typename scalar_t> struct TwiddleWriter {
  CudaAcsr<scalar_t, 5> twiddle_a;

  TwiddleWriter(torch::Tensor twiddle):
    twiddle_a(twiddle.packed_accessor32<scalar_t, 5, at::RestrictPtrTraits>()) {}

  template<int nsteps, bool increasing_stride, typename accscalar_t>
  __device__ __forceinline__ void save_max5(accscalar_t s_twiddle[nsteps][2][1 << nsteps],
                                            int input_idx_start_bit, int low_bits, int high_bits) {
    constexpr int span = 1 << nsteps;
    const int s = blockIdx.y + gridDim.y * blockIdx.z;  // For conv2d butterfly as well
    for (int t = threadIdx.x + threadIdx.y * blockDim.x; t < nsteps * (span / 2); t += blockDim.x * blockDim.y) {
      const int step = t / (span / 2);
      const int s_twiddle_stride = 1 << (increasing_stride ? step : nsteps - 1 - step);
      const int remainder = t % (span / 2);
      const int low_order_bits = remainder & (s_twiddle_stride - 1);
      const int s_idx = 2 * (remainder - low_order_bits) + low_order_bits;
      const int idx = (high_bits >> 1) | (remainder << input_idx_start_bit) | low_bits;
      gpuAtomicAdd(&twiddle_a[s][step][idx][0][0], (scalar_t) s_twiddle[step][0][s_idx]);
      gpuAtomicAdd(&twiddle_a[s][step][idx][0][1], (scalar_t) s_twiddle[step][1][s_idx]);
      gpuAtomicAdd(&twiddle_a[s][step][idx][1][0], (scalar_t) s_twiddle[step][1][s_idx + s_twiddle_stride]);
      gpuAtomicAdd(&twiddle_a[s][step][idx][1][1], (scalar_t) s_twiddle[step][0][s_idx + s_twiddle_stride]);
    }
  }

};

template <int nsteps, bool increasing_stride, int items_per_thread,
            int mult_per_warp=1, typename scalar_t>
__device__ __forceinline__ void b_untied_forward_shared_twiddle(const scalar_t s_twiddle[nsteps][2][1 << nsteps],
                                                                scalar_t input_val[mult_per_warp][items_per_thread],
                                                                const int t_idx) {
  #pragma unroll
  for (int step = 0; step < nsteps; step++) {
    int log_stride = increasing_stride ? step : nsteps - 1 - step;
    if (log_stride < 5) {
      int lane_mask = 1 << log_stride;
      #pragma unroll
      for (int mult = 0; mult < mult_per_warp; mult++) {
        const scalar_t twiddle_val[2] = {s_twiddle[step][0][mult * warpSize + t_idx],
                                         s_twiddle[step][1][mult * warpSize + t_idx]};
        #pragma unroll
        for (int item = 0; item < items_per_thread; item++) {
          scalar_t input_val_other = __shfl_xor_sync(FULL_MASK, input_val[mult][item], lane_mask);
          input_val[mult][item] = twiddle_val[0] * input_val[mult][item] + twiddle_val[1] * input_val_other;
        }
      }
    } else {
      int mult_stride = 1 << (log_stride - 5);
      #pragma unroll
      for (int m = 0; m < mult_per_warp / 2; m++) {
        int low_order_bits = m & (mult_stride - 1);  // int low_order_bits = m % mult_stride;
        int mult = 2 * (m - low_order_bits) + low_order_bits;
        const scalar_t twiddle_val[2][2]
          = {{s_twiddle[step][0][mult * warpSize + t_idx],
              s_twiddle[step][1][mult * warpSize + t_idx]},
             {s_twiddle[step][0][(mult + mult_stride) * warpSize + t_idx],
              s_twiddle[step][1][(mult + mult_stride) * warpSize + t_idx]}};
        #pragma unroll
        for (int item = 0; item < items_per_thread; item++) {
          scalar_t inputs[2] = {input_val[mult][item], input_val[mult + mult_stride][item]};
          input_val[mult][item] = twiddle_val[0][0] * inputs[0] + twiddle_val[0][1] * inputs[1];
          // The order of twiddle[1] is swapped by design
          input_val[mult + mult_stride][item] = twiddle_val[1][1] * inputs[0] + twiddle_val[1][0] * inputs[1];
        }
      }
    }
  }
}

template <int nsteps, bool increasing_stride,
            int items_per_thread=ITEMS_PER_THREAD_FORWARD_MAX5[nsteps - 1],
            int min_blocks_per_mp=MIN_BLOCKS_PER_MP_FORWARD[nsteps - 1],
            typename scalar_t>
C10_LAUNCH_BOUNDS_2(MAX5_FORWARD_BLOCK_SIZE, min_blocks_per_mp)
__global__ void butterfly_multiply_untied_forward_max5_fast_cuda_kernel(TwiddleReader<scalar_t> twiddle_reader,
                                                                        InputReader<scalar_t> input_reader,
                                                                        OutputWriter<scalar_t> output_writer,
                                                                        int log_n,
                                                                        int input_idx_start_bit) {
  constexpr int span = 1 << nsteps;
  constexpr int mult_per_warp = span > WARP_SIZE ? span / WARP_SIZE : 1;
  // We just want a shared array:
  // __shared__ scalar_t s_twiddle[nsteps][2][span];
  // But it doesn't work for complex: https://github.com/pytorch/pytorch/issues/39270.
  // So we declare a char array and cast it.
  // The casting is subtle: https://stackoverflow.com/questions/12692310/convert-array-to-two-dimensional-array-by-pointer
  __shared__ char s_twiddle_char[nsteps][2][span * sizeof(scalar_t)];
  scalar_t (*s_twiddle)[2][span] = (scalar_t (*)[2][span])&s_twiddle_char[0][0][0];
  scalar_t input_val[mult_per_warp][items_per_thread];
  const int t_idx = threadIdx.x;
  const int batch_idx = (threadIdx.y + (blockIdx.x >> (log_n - nsteps)) * blockDim.y) * items_per_thread;
  const int remaining_input_idx = blockIdx.x & ((1 << (log_n - nsteps)) - 1);
  const int low_bits = remaining_input_idx & ((1 << input_idx_start_bit) - 1);
  const int high_bits = (remaining_input_idx >> input_idx_start_bit) << (input_idx_start_bit + nsteps);
  // All threads with the same t_idx should have the same input_idx
  const int input_idx = high_bits | (t_idx << input_idx_start_bit) | low_bits;
  const int input_idx_stride = (1 << input_idx_start_bit) * warpSize;
  twiddle_reader.load_max5<nsteps, increasing_stride>(s_twiddle, input_idx_start_bit, low_bits, high_bits);
  input_reader.load_max5<items_per_thread, mult_per_warp>(input_val, batch_idx, input_idx, input_idx_stride);
  __syncthreads();
  b_untied_forward_shared_twiddle<nsteps, increasing_stride, items_per_thread, mult_per_warp>(s_twiddle, input_val, t_idx);
  output_writer.save_max5<items_per_thread, mult_per_warp>(input_val, batch_idx, input_idx, input_idx_stride);
}

std::vector<int> butterfly_max5_plan(const int log_n, const int nblocks, const int max_nsteps, const bool increasing_stride) {
  const int niters = div_up(log_n, max_nsteps);
  const int niters_total = niters * nblocks;
  const int nsteps = div_up(log_n, niters);
  const int nsteps_remainder = log_n - (niters - 1) * nsteps;
  std::vector<int> bit_milestones;
  bit_milestones.reserve(niters_total + 1);
  auto push_strides = [log_n, nsteps, nsteps_remainder](std::vector<int>& milestones, bool increasing) {
    if (increasing) {
      for (int i = nsteps_remainder; i <= log_n; i += nsteps) {
        milestones.push_back(i);
      }
    } else {
      for (int i = log_n - nsteps; i > 0; i -= nsteps) {
        milestones.push_back(i);
      }
      milestones.push_back(0);
    }
  };
  // For each block, bit_milestones has niters + 1 elements the form
  // [0, nsteps_remainder, nsteps_remainder + nsteps, ..., log_n]
  // if increasing stride. Otherwise it's the reverse.
  bit_milestones.push_back(increasing_stride ? 0 : log_n);
  for (int block = 0; block < nblocks; block++) {
    bool cur_increasing_stride = increasing_stride != bool(block % 2);
    push_strides(bit_milestones, cur_increasing_stride);
  }
  return bit_milestones;
}

using namespace torch::indexing;

torch::Tensor butterfly_multiply_fw_cuda(const torch::Tensor twiddle,
                                         const torch::Tensor input,
                                         bool increasing_stride,
                                         int output_size) {
  int batch_size = input.size(0);
  const int nstacks = input.size(1);
  const int nblocks = twiddle.size(1);
  const int log_n = twiddle.size(2);
  const int n = 1 << log_n;
  auto output = torch::empty({batch_size, nstacks, n},
                             torch::dtype(input.dtype()).device(input.device()));
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(input.scalar_type(), "butterfly_multiply_fw_cuda", [&] {
    constexpr int MAXSTEP_FW = maxstep<scalar_t>::MAXSTEP_FW;
    const std::vector<int> bit_milestones = butterfly_max5_plan(log_n, nblocks, MAXSTEP_FW, increasing_stride);
    const int niters = bit_milestones.size() - 1;
    const auto input_a = input.packed_accessor32<scalar_t, 3, at::RestrictPtrTraits>();
    auto output_a = output.packed_accessor32<scalar_t, 3, at::RestrictPtrTraits>();
    int twiddle_block_idx = 0;
    int twiddle_idx_start = 0;
    for (int iter = 0; iter < niters; iter++) {
      const InputReader<scalar_t> input_reader(iter == 0 ? input_a : output_a);
      // Reduce size of output because we only want the first @output_size elements
      OutputWriter<scalar_t> output_writer = iter == niters - 1 ?
        OutputWriter<scalar_t>(output.index({Slice(), Slice(), Slice(None, output_size)})) : OutputWriter<scalar_t>(output_a);
      const bool increasing_stride_this_iter = bit_milestones[iter] <= bit_milestones[iter + 1];
      const int start_bit = increasing_stride_this_iter ? bit_milestones[iter] : bit_milestones[iter + 1];
      const int nsteps = abs(bit_milestones[iter + 1] - bit_milestones[iter]);
      const int span = 1 << nsteps;
      const int n_div_span = 1 << (log_n - nsteps);  // = n / span
      const int block_x = min(span, WARP_SIZE);
      const int max_block_y = MAX5_FORWARD_BLOCK_SIZE / block_x;
      dim3 block(block_x, min(max_block_y, div_up(batch_size, ITEMS_PER_THREAD_FORWARD_MAX5[nsteps - 1])));
      // grid.x must be at least n / span
      dim3 grid(div_up(batch_size, ITEMS_PER_THREAD_FORWARD_MAX5[nsteps - 1] * block.y) * n_div_span, 1, nstacks);
      // https://pytorch.org/cppdocs/notes/tensor_indexing.html
      // twiddle[:, twiddle_block_idx, twiddle_idx_start:twiddle_idx_start + nsteps]
      const TwiddleReader<scalar_t> twiddle_reader(twiddle.index(
          {Slice(), twiddle_block_idx, Slice(twiddle_idx_start, twiddle_idx_start + nsteps)}));
      // Template-metaprogramming hackery: we want nsteps as a constexpr so we can dispatch on it.
      // Which means we want templated lambdas. But templated lambda isn't available until C++20.
      // Generic lambdas (starting C++14) allows a kind of templated lambda, but only template on type (and not int).
      // So we have to construct std::integral_constant to encode the int value as a type.
      auto launch = [increasing_stride_this_iter, &grid, &block, &stream, &twiddle_reader, &input_reader, &output_writer,
                     log_n, start_bit] (auto dummy) {
      // I can't capture all with [&] for some reason (nvcc complains that start_bit isn't captured).
        constexpr int nsteps_val = decltype(dummy)::value;
        increasing_stride_this_iter ? butterfly_multiply_untied_forward_max5_fast_cuda_kernel<nsteps_val, true>
          <<<grid, block, 0, stream>>>(twiddle_reader, input_reader, output_writer, log_n, start_bit)
          : butterfly_multiply_untied_forward_max5_fast_cuda_kernel<nsteps_val, false>
          <<<grid, block, 0, stream>>>(twiddle_reader, input_reader, output_writer, log_n, start_bit);
      };
      Dispatch<MAXSTEP_FW, decltype(launch)>::call(launch, nsteps);
      twiddle_idx_start += nsteps;
      if (twiddle_idx_start >= log_n) {
        twiddle_idx_start = 0;
        twiddle_block_idx++;
      }
    }
  });
  TORCH_CHECK(cudaGetLastError() == cudaSuccess,
     "butterfly_multiply_fw_cuda failed with error code ",
     cudaGetLastError());
  if (output_size < n) {
    output = output.index({Slice(), Slice(), Slice(None, output_size)});
  }
  return output;
}

template <int nsteps, bool increasing_stride, int items_per_thread, int mult_per_warp=1,
            typename scalar_t, typename accscalar_t=at::acc_type<scalar_t, true>>
__device__ __forceinline__ void b_untied_forward_backward_shared_twiddle(const scalar_t s_twiddle[nsteps][2][1 << nsteps],
                                                                         accscalar_t s_d_twiddle[nsteps][2][1 << nsteps],
                                                                         scalar_t input_val[nsteps][mult_per_warp][items_per_thread],
                                                                         scalar_t grad_val[mult_per_warp][items_per_thread],
                                                                         int t_idx) {
  // Forward pass
  #pragma unroll
  for (int step = 0; step < nsteps - 1; step++) {  // Don't need input for the last step
    int log_stride = increasing_stride ? step : nsteps - 1 - step;
    int lane_mask = 1 << log_stride;
    #pragma unroll
    for (int mult = 0; mult < mult_per_warp; mult++) {
      const scalar_t twiddle_val[2] = {s_twiddle[step][0][mult * warpSize + t_idx],
                                       s_twiddle[step][1][mult * warpSize + t_idx]};
      #pragma unroll
      for (int item = 0; item < items_per_thread; item++) {
        scalar_t input_val_other = log_stride < 5 ?
            __shfl_xor_sync(FULL_MASK, input_val[step][mult][item], lane_mask)
          : input_val[step][mult ^ (1 << (log_stride - 5))][item];
        input_val[step + 1][mult][item] = twiddle_val[0] * input_val[step][mult][item] + twiddle_val[1] * input_val_other;
      }
    }
  }
  // Backward pass
  #pragma unroll
  for (int step = nsteps - 1; step >= 0; step--) {
    int log_stride = increasing_stride ? step : nsteps - 1 - step;
    int lane_mask = 1 << log_stride;
    #pragma unroll
    for (int mult = 0; mult < mult_per_warp; mult++) {
      const scalar_t twiddle_val[2] = {s_twiddle[step][0][mult * warpSize + t_idx],
                                       s_twiddle[step][1][mult * warpSize + t_idx]};
      accscalar_t d_twiddle_val[2] = {0};
      #pragma unroll
      for (int item = 0; item < items_per_thread; item++) {
        d_twiddle_val[0] += grad_val[mult][item] * conj_wrapper(input_val[step][mult][item]);
        scalar_t input_val_other = log_stride < 5 ?
            __shfl_xor_sync(FULL_MASK, input_val[step][mult][item], lane_mask)
          : input_val[step][mult ^ (1 << (log_stride - 5))][item];
        d_twiddle_val[1] += grad_val[mult][item] * conj_wrapper(input_val_other);
        if (log_stride < 5) {
          grad_val[mult][item] = conj_wrapper(twiddle_val[0]) * grad_val[mult][item]
            + __shfl_xor_sync(FULL_MASK, conj_wrapper(twiddle_val[1]) * grad_val[mult][item], lane_mask);
        }
      }
      gpuAtomicAdd(&s_d_twiddle[step][0][mult * warpSize + t_idx], d_twiddle_val[0]);
      gpuAtomicAdd(&s_d_twiddle[step][1][mult * warpSize + t_idx], d_twiddle_val[1]);
    }
    if (log_stride >= 5) {
      int mult_stride = 1 << (log_stride - 5);
      #pragma unroll
      for (int m = 0; m < mult_per_warp / 2; m++) {
        int low_order_bits = m & (mult_stride - 1);  // int low_order_bits = m % mult_stride;
        int mult = 2 * (m - low_order_bits) + low_order_bits;
        const scalar_t twiddle_val[2][2]
          = {{s_twiddle[step][0][mult * warpSize + t_idx],
              s_twiddle[step][1][mult * warpSize + t_idx]},
             {s_twiddle[step][0][(mult + mult_stride) * warpSize + t_idx],
              s_twiddle[step][1][(mult + mult_stride) * warpSize + t_idx]}};
        #pragma unroll
        for (int item = 0; item < items_per_thread; item++) {
          scalar_t grads[2] = {grad_val[mult][item], grad_val[mult + mult_stride][item]};
          // The order of twiddle[1] is swapped by design
          grad_val[mult][item] = conj_wrapper(twiddle_val[0][0]) * grads[0] + conj_wrapper(twiddle_val[1][1]) * grads[1];
          grad_val[mult + mult_stride][item] = conj_wrapper(twiddle_val[0][1]) * grads[0] + conj_wrapper(twiddle_val[1][0]) * grads[1];
        }
      }
    }
  }
}

template <int nsteps, bool increasing_stride,
            int items_per_thread=ITEMS_PER_THREAD_BACKWARD_MAX5[nsteps - 1],
            int min_blocks_per_mp=MIN_BLOCKS_PER_MP_BACKWARD[nsteps - 1],
            typename scalar_t>
C10_LAUNCH_BOUNDS_2(MAX5_BACKWARD_BLOCK_SIZE, min_blocks_per_mp)
__global__ void butterfly_multiply_untied_forward_backward_max5_fast_cuda_kernel(TwiddleReader<scalar_t> twiddle_reader,
                                                                                 InputReader<scalar_t> input_reader,
                                                                                 InputReader<scalar_t> grad_reader,
                                                                                 TwiddleWriter<scalar_t> d_twiddle_writer,
                                                                                 OutputWriter<scalar_t> d_input_writer,
                                                                                 int log_n,
                                                                                 int input_idx_start_bit) {
  using accscalar_t = at::acc_type<scalar_t, true>;
  constexpr int span = 1 << nsteps;
  constexpr int mult_per_warp = span > WARP_SIZE ? span / WARP_SIZE : 1;
  // __shared__ scalar_t s_twiddle[nsteps][2][span];
  __shared__ char s_twiddle_char[nsteps][2][span * sizeof(scalar_t)];
  scalar_t (*s_twiddle)[2][span] = (scalar_t (*)[2][span])&s_twiddle_char[0][0][0];
  // __shared__ accscalar_t s_d_twiddle[nsteps][2][span];
  __shared__ char s_d_twiddle_char[nsteps][2][span * sizeof(accscalar_t)];
  accscalar_t (*s_d_twiddle)[2][span] = (accscalar_t (*)[2][span])&s_d_twiddle_char[0][0][0];
  initalize_shared_mem<accscalar_t>((scalar_t *)&s_d_twiddle[0][0][0], nsteps * 2 * span);
  scalar_t input_val[nsteps][mult_per_warp][items_per_thread];
  scalar_t grad_val[mult_per_warp][items_per_thread];
  const int t_idx = threadIdx.x;
  const int batch_idx = (threadIdx.y + (blockIdx.x >> (log_n - nsteps)) * blockDim.y) * items_per_thread;
  const int remaining_input_idx = blockIdx.x & ((1 << (log_n - nsteps)) - 1);
  const int low_bits = remaining_input_idx & ((1 << input_idx_start_bit) - 1);
  const int high_bits = (remaining_input_idx >> input_idx_start_bit) << (input_idx_start_bit + nsteps);
  // All threads with the same t_idx should have the same input_idx
  const int input_idx = high_bits | (t_idx << input_idx_start_bit) | low_bits;
  const int input_idx_stride = (1 << input_idx_start_bit) * warpSize;
  twiddle_reader.load_max5<nsteps, increasing_stride>(s_twiddle, input_idx_start_bit, low_bits, high_bits);
  input_reader.load_max5<items_per_thread, mult_per_warp>(input_val[0], batch_idx, input_idx, input_idx_stride);
  __syncthreads();
  grad_reader.load_max5<items_per_thread, mult_per_warp>(grad_val, batch_idx, input_idx, input_idx_stride);
  b_untied_forward_backward_shared_twiddle<nsteps, increasing_stride, items_per_thread, mult_per_warp>
    (s_twiddle, s_d_twiddle, input_val, grad_val, t_idx);
  d_input_writer.save_max5<items_per_thread, mult_per_warp>(grad_val, batch_idx, input_idx, input_idx_stride);
  __syncthreads();
  d_twiddle_writer.save_max5<nsteps, increasing_stride>(s_d_twiddle, input_idx_start_bit, low_bits, high_bits);
}

std::tuple<torch::Tensor, torch::Tensor>
  butterfly_multiply_bw_cuda(const torch::Tensor twiddle,
                             const torch::Tensor input,
                             const torch::Tensor grad,
                             bool increasing_stride) {
  int batch_size = input.size(0);
  const int nstacks = input.size(1);
  const int nblocks = twiddle.size(1);
  const int log_n = twiddle.size(2);
  const int n = 1 << log_n;
  const int input_size = input.size(2);
  const int output_size = grad.size(2);
  auto d_input = torch::empty({batch_size, nstacks, std::max(input_size, n)},
                   torch::dtype(input.dtype()).device(input.device()));
  auto d_twiddle = torch::zeros_like(twiddle);
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(input.scalar_type(), "butterfly_multiply_bw_cuda", [&] {
    constexpr int MAXSTEP_BW = maxstep<scalar_t>::MAXSTEP_BW;
    const std::vector<int> bit_milestones = butterfly_max5_plan(log_n, nblocks, MAXSTEP_BW, increasing_stride);
    const int niters = bit_milestones.size() - 1;
    auto intermediate_storage = torch::empty({niters - 1, batch_size, nstacks, n},
                                              torch::dtype(input.dtype()).device(input.device()));
    // Forward pass
    int twiddle_block_idx = 0;
    int twiddle_idx_start = 0;
    // Don't need the very last output
    for (int iter = 0; iter < niters - 1; iter++) {
      // We want to index on intermediate_storage_a but PackedTensorAccessor indexing only works on device code
      // So we have the index on the tensor itself, not on the accessor.
      const InputReader<scalar_t> input_reader(iter == 0 ? input : intermediate_storage[iter - 1]);
      OutputWriter<scalar_t> output_writer(intermediate_storage[iter]);
      const bool increasing_stride_this_iter = bit_milestones[iter] <= bit_milestones[iter + 1];
      const int start_bit = increasing_stride_this_iter ? bit_milestones[iter] : bit_milestones[iter + 1];
      const int nsteps = abs(bit_milestones[iter + 1] - bit_milestones[iter]);
      const int span = 1 << nsteps;
      const int n_div_span = 1 << (log_n - nsteps);  // = n / span
      const int block_x = min(span, WARP_SIZE);
      const int max_block_y = MAX5_FORWARD_BLOCK_SIZE / block_x;
      dim3 block(block_x, min(max_block_y, div_up(batch_size, ITEMS_PER_THREAD_FORWARD_MAX5[nsteps - 1])));
      // grid.x must be at least n / span
      dim3 grid(div_up(batch_size, ITEMS_PER_THREAD_FORWARD_MAX5[nsteps - 1] * block.y) * n_div_span, 1, nstacks);
      // twiddle[:, twiddle_block_idx, twiddle_idx_start:twiddle_idx_start + nsteps]
      const TwiddleReader<scalar_t> twiddle_reader(twiddle.index(
          {Slice(), twiddle_block_idx, Slice(twiddle_idx_start, twiddle_idx_start + nsteps)}));
      auto launch = [increasing_stride_this_iter, &grid, &block, &stream, &twiddle_reader, &input_reader, &output_writer,
                     log_n, start_bit] (auto dummy) {
        constexpr int nsteps_val = decltype(dummy)::value;
        increasing_stride_this_iter ? butterfly_multiply_untied_forward_max5_fast_cuda_kernel<nsteps_val, true>
          <<<grid, block, 0, stream>>>(twiddle_reader, input_reader, output_writer, log_n, start_bit)
          : butterfly_multiply_untied_forward_max5_fast_cuda_kernel<nsteps_val, false>
          <<<grid, block, 0, stream>>>(twiddle_reader, input_reader, output_writer, log_n, start_bit);
      };
      Dispatch<MAXSTEP_BW, decltype(launch)>::call(launch, nsteps);
      twiddle_idx_start += nsteps;
      if (twiddle_idx_start >= log_n) {
        twiddle_idx_start = 0;
        twiddle_block_idx++;
      }
    }
    // Backward pass
    const auto grad_a = grad.packed_accessor32<scalar_t, 3, at::RestrictPtrTraits>();
    auto d_input_a = d_input.packed_accessor32<scalar_t, 3, at::RestrictPtrTraits>();
    twiddle_block_idx = nblocks - 1;
    twiddle_idx_start = log_n;
    for (int iter = niters - 1; iter >= 0; iter--) {
      // Reduce size of d_input_a because we only want the first @input_size elements
      OutputWriter<scalar_t> d_input_writer = iter == 0 ?
        OutputWriter<scalar_t>(d_input.index({Slice(), Slice(), Slice(None, input_size)})) : OutputWriter<scalar_t>(d_input_a);
      const InputReader<scalar_t> input_reader(iter == 0 ? input : intermediate_storage[iter - 1]);
      const InputReader<scalar_t> grad_reader(iter == niters - 1 ? grad_a : d_input_a);
      const bool increasing_stride_this_iter = bit_milestones[iter] <= bit_milestones[iter + 1];
      const int start_bit = increasing_stride_this_iter ? bit_milestones[iter] : bit_milestones[iter + 1];
      const int nsteps = abs(bit_milestones[iter + 1] - bit_milestones[iter]);
      twiddle_idx_start -= nsteps;
      if (twiddle_idx_start < 0) {
        twiddle_block_idx--;
        twiddle_idx_start += log_n;
      }
      const int span = 1 << nsteps;
      const int n_div_span = 1 << (log_n - nsteps);  // = n / span
      const int block_x = min(span, WARP_SIZE);
      const int max_block_y = MAX5_BACKWARD_BLOCK_SIZE / block_x;
      dim3 block(block_x, min(max_block_y, div_up(batch_size, ITEMS_PER_THREAD_BACKWARD_MAX5[nsteps - 1])));
      // grid.x must be at least n / span
      dim3 grid(div_up(batch_size, ITEMS_PER_THREAD_BACKWARD_MAX5[nsteps - 1] * block.y) * n_div_span, 1, nstacks);
      // twiddle[:, twiddle_block_idx, twiddle_idx_start:twiddle_idx_start + nsteps]
      const TwiddleReader<scalar_t> twiddle_reader(twiddle.index(
          {Slice(), twiddle_block_idx, Slice(twiddle_idx_start, twiddle_idx_start + nsteps)}));
      TwiddleWriter<scalar_t> d_twiddle_writer(d_twiddle.index(
          {Slice(), twiddle_block_idx, Slice(twiddle_idx_start, twiddle_idx_start + nsteps)}));
      auto launch = [increasing_stride_this_iter, &grid, &block, &stream, &twiddle_reader, &input_reader,
                     &grad_reader, &d_twiddle_writer, &d_input_writer, log_n, start_bit] (auto dummy) {
        constexpr int nsteps_val = decltype(dummy)::value;
        increasing_stride_this_iter ? butterfly_multiply_untied_forward_backward_max5_fast_cuda_kernel<nsteps_val, true>
          <<<grid, block, 0, stream>>>(twiddle_reader, input_reader, grad_reader, d_twiddle_writer, d_input_writer,
                                       log_n, start_bit)
          : butterfly_multiply_untied_forward_backward_max5_fast_cuda_kernel<nsteps_val, false>
          <<<grid, block, 0, stream>>>(twiddle_reader, input_reader, grad_reader, d_twiddle_writer, d_input_writer,
                                       log_n, start_bit);
      };
      Dispatch<MAXSTEP_BW, decltype(launch)>::call(launch, nsteps);
    }
  });
  TORCH_CHECK(cudaGetLastError() == cudaSuccess,
     "butterfly_multiply_bw_cuda failed with error code ",
     cudaGetLastError());
  if (input_size < n) {
    d_input = d_input.index({Slice(), Slice(), Slice(None, input_size)});
  } else if (input_size > n) {
    d_input.index({Slice(), Slice(), Slice(n, None)}).fill_(0.0);
  }
  return std::make_tuple(d_twiddle, d_input);
}
