#include <torch/extension.h>

// This isn't implemented for c10::complex yet in Pytorch
template <typename scalar_t>
static __device__ __forceinline__ c10::complex<scalar_t>
__shfl_xor_sync(unsigned int mask, c10::complex<scalar_t> value,
                unsigned int laneMask, int width = warpSize) {
  return c10::complex<scalar_t>(
      __shfl_xor_sync(mask, value.real_, laneMask, width),
      __shfl_xor_sync(mask, value.imag_, laneMask, width));
}
