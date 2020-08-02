#include <torch/extension.h>
#include <ATen/AccumulateType.h>  // For at::acc_type

// This isn't implemented for c10::complex yet in Pytorch
template <typename scalar_t>
static __device__ __forceinline__ c10::complex<scalar_t>
__shfl_xor_sync(unsigned int mask, c10::complex<scalar_t> value,
                unsigned int laneMask, int width = warpSize) {
  return c10::complex<scalar_t>(
      __shfl_xor_sync(mask, value.real_, laneMask, width),
      __shfl_xor_sync(mask, value.imag_, laneMask, width));
}

// We manually overload conj because std::conj does not work types other than
// c10::complex.
template <typename scalar_t>
__host__ __device__ static inline scalar_t conj_wrapper(scalar_t v) {
  return v;
}

template <typename T>
__host__ __device__ static inline c10::complex<T> conj_wrapper(c10::complex<T> v) {
  return std::conj(v);
}

// This isn't defined yet (pytorch 1.6) for c10::complex with is_cuda=True
// until https://github.com/pytorch/pytorch/commit/324c18fcad579b1afa63ae45528bf598ba8ec4ca
// TODO: remove if compiling with pytorch-nightly or once pytorch 1.7 is released.
namespace at {
template <> struct AccumulateType<c10::complex<float>, true> { using type = c10::complex<float>; };
template <> struct AccumulateType<c10::complex<double>, true> { using type = c10::complex<double>; };
}