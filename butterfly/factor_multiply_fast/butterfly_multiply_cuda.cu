#include <stdio.h>
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THCAtomics.cuh>  // For atomicAdd on Half
#include <c10/macros/Macros.h>  // For __launch_bounds__
#include <thrust/complex.h>
#include <thrust/pair.h>
#include <thrust/tuple.h>
#include "map.h"  // For the MAP macro, i.e. for_each over the arguments

#define BFLY_BENCHMARK false

#define thc_cos std::cos
#define thc_sin std::sin

#define FULL_MASK 0xffffffff

#define MIN_MACRO(x, y) (((x) <= (y)) ? (x) : (y))

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
  static constexpr int SMEM_PER_MP = 96 * (1 << 10);
#else
  static constexpr int SMEM_PER_MP = 64 * (1 << 10);
#endif
// static constexpr int SMEM_PER_MP = 64 * (1 << 10);
static constexpr int MAX_SMEM_PER_BLOCK = 48 * (1 << 10);
static constexpr int MAX_BLOCK_SIZE = 1024;
// static constexpr int WORK_PER_THREAD = 16;
// static constexpr int ELEMENTARY_SIZE = MAX_BLOCK_SIZE / 2;
// static constexpr int MAX_N_FACTORS = 10;
static constexpr int ITEMS_PER_THREAD_FORWARD[14] = {4, 4, 4, 4, 4, 4, 4, 8, 8, 8, 13, 10, 4, 4};
static constexpr int ITEMS_PER_THREAD_BACKWARD[14] = {16, 16, 16, 16, 16, 16, 16, 16, 16, 8, 8, 8, 8, 8};
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

template <typename T, size_t N>
using CudaAcsr = at::PackedTensorAccessor<T, N, at::RestrictPtrTraits, int32_t>;

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

template <typename scalar_t>
static __device__  __forceinline__
void atomicAdd(thrust::complex<scalar_t> *address,
               thrust::complex<scalar_t> val) {
  atomicAdd((scalar_t *)address, val.real());
  atomicAdd((scalar_t *)address + 1, val.imag());
}

template <typename scalar_t>
static __device__  __forceinline__
thrust::complex<scalar_t> __shfl_down_sync(unsigned int mask,
                                           thrust::complex<scalar_t> value,
                                           unsigned int delta,
                                           int width = warpSize) {
  return thrust::complex<scalar_t>(__shfl_down_sync(mask, value.real(), delta, width),
                                   __shfl_down_sync(mask, value.imag(), delta, width));
}

template<typename scalar_t>
struct InputReader {
  const CudaAcsr<scalar_t, 3> input_a;
  const int batch_size;
  InputReader(const at::Tensor input):
    input_a(input.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>()),
      batch_size(input.size(0)) {}

  template<int items_per_thread, int mult_per_warp=1>
  __device__ __forceinline__ void load(scalar_t input_val[mult_per_warp][items_per_thread],
                                       int idx) {
    #pragma unroll
    for (int mult = 0; mult < mult_per_warp; mult++) {
      int i = mult * warpSize + idx;
      const int s = blockIdx.z;
      #pragma unroll
      for (int item = 0; item < items_per_thread; item++){
        const int b = blockIdx.x * items_per_thread + item;
        input_val[mult][item] = b < batch_size ? input_a[b][s][i] : 0;
      }
    }
  }

};

template<typename scalar_t>
struct OutputWriter {
  CudaAcsr<scalar_t, 3> output_a;
  const int batch_size;
  OutputWriter(at::Tensor output):
    output_a(output.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>()),
      batch_size(output.size(0)) {}

  template<int items_per_thread, int mult_per_warp=1>
  __device__ __forceinline__ void save(scalar_t output_val[mult_per_warp][items_per_thread],
                                       int idx) {
    #pragma unroll
    for (int mult = 0; mult < mult_per_warp; mult++) {
      int i = mult * warpSize + idx;
      const int s = blockIdx.z;
      #pragma unroll
      for (int item = 0; item < items_per_thread; item++){
        const int b = blockIdx.x * items_per_thread + item;
        if (b < batch_size) {
          output_a[b][s][i] = output_val[mult][item];
        }
      }
    }
  }

};

template<typename scalar_t>
struct IntermediateStorage {
  CudaAcsr<scalar_t, 4> storage_a;
  IntermediateStorage(const at::Tensor storage):
    storage_a(storage.packed_accessor<scalar_t, 4, at::RestrictPtrTraits, int32_t>()) {}

  template<int items_per_thread, int mult_per_warp=1>
  __device__ __forceinline__ void save(scalar_t output_val[mult_per_warp][items_per_thread],
                                       int idx, int step) {
    #pragma unroll
    for (int mult = 0; mult < mult_per_warp; mult++) {
      int i = mult * warpSize + idx;
      const int s = blockIdx.z;
      #pragma unroll
      for (int item = 0; item < items_per_thread; item++){
        const int b = blockIdx.x * items_per_thread + item;
        storage_a[step][b][s][i] = output_val[mult][item];
      }
    }
  }

  template<int items_per_thread, int mult_per_warp=1>
  __device__ __forceinline__ void load(scalar_t input_val[mult_per_warp][items_per_thread],
                                       int idx, int step) {
    #pragma unroll
    for (int mult = 0; mult < mult_per_warp; mult++) {
      int i = mult * warpSize + idx;
      const int s = blockIdx.z;
      #pragma unroll
      for (int item = 0; item < items_per_thread; item++){
        const int b = blockIdx.x * items_per_thread + item;
        input_val[mult][item] = storage_a[step][b][s][i];
      }
    }
  }

};

template <int items_per_thread, int mult_per_warp=1,
            int smem_per_thread=items_per_thread, typename scalar_t>
__device__ __forceinline__ void block_exchange(scalar_t *temp_storage,
                                               scalar_t values[mult_per_warp][items_per_thread],
                                               int thread_idx_1,
                                               int thread_idx_2,
                                               int nthreads) {
  constexpr int nsteps = div_up_const(items_per_thread, smem_per_thread);
  // TODO: combine mult_per_warp and iterms_per_thread, i.e. 2D -> 1D, to reduce number of syncthreads.
  #pragma unroll
  for (int mult = 0; mult < mult_per_warp; mult++) {
    #pragma unroll
    for (int i = 0; i < nsteps; i++) {
      if ((i > 0) || (mult > 0)) {
        __syncthreads();
      }
      #pragma unroll
      for (int item = 0; (item < smem_per_thread) && (i * smem_per_thread + item < items_per_thread); item++) {
        temp_storage[thread_idx_1 + item * nthreads] = values[mult][i * smem_per_thread + item];
      }
      __syncthreads();
      #pragma unroll
      for (int item = 0; (item < smem_per_thread) && (i * smem_per_thread + item < items_per_thread); item++) {
        values[mult][i * smem_per_thread + item] = temp_storage[thread_idx_2 + item * nthreads];
      }
    }
  }
}

template <int nsteps, bool increasing_stride, int items_per_thread,
            int mult_per_warp=1, typename scalar_t>
__device__ __forceinline__ void b_untied_forward(const CudaAcsr<scalar_t, 4> twiddle_a,
                                                 scalar_t input_val[mult_per_warp][items_per_thread],
                                                 int twiddle_idx_start,
                                                 int input_idx) {
  const int s = blockIdx.y + gridDim.y * blockIdx.z;  // For conv2d butterfly as well
  #pragma unroll
  // TODO: for loop over mult first instead of step first,
  // will have to split into 2 parts: intra-thread and intra-warp.
  for (int step = 0; step < nsteps; step++) {
    int log_stride = increasing_stride ? step : nsteps - 1 - step;
    int twiddle_idx = twiddle_idx_start + step;
    if (log_stride < 5) {
      int lane_mask = 1 << log_stride;
      #pragma unroll
      for (int mult = 0; mult < mult_per_warp; mult++) {
        // TODO: make num thread per warp an input argument
        const scalar_t twiddle_val[2] = {twiddle_a[s][twiddle_idx][0][mult * warpSize + input_idx],
                                         twiddle_a[s][twiddle_idx][1][mult * warpSize + input_idx]};
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
          = {{twiddle_a[s][twiddle_idx][0][mult * warpSize + input_idx],
              twiddle_a[s][twiddle_idx][1][mult * warpSize + input_idx]},
             {twiddle_a[s][twiddle_idx][0][(mult + mult_stride) * warpSize + input_idx],
              twiddle_a[s][twiddle_idx][1][(mult + mult_stride) * warpSize + input_idx]}};
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

template <int log_n, bool increasing_stride,
            int items_per_thread=ITEMS_PER_THREAD_FORWARD[log_n - 1],
            int min_blocks_per_mp=MIN_BLOCKS_PER_MP_FORWARD[log_n - 1],
            int max_smem_per_thread=items_per_thread, typename scalar_t>
// C10_LAUNCH_BOUNDS_2 supposedly takes min(1 << log_n, 1024)
// https://github.com/pytorch/pytorch/blob/v1.1.0/c10/macros/Macros.h
// However, it doesn't seem to work correctly so I have to take min explicitly.
C10_LAUNCH_BOUNDS_2(MIN_MACRO(1 << log_n, MAX_BLOCK_SIZE), min_blocks_per_mp)
__global__ void butterfly_multiply_untied_forward_fast_cuda_kernel(const CudaAcsr<scalar_t, 4> twiddle_a,
                                                                   InputReader<scalar_t> input_reader,
                                                                   OutputWriter<scalar_t> output_writer,
                                                                   int batch_size) {
  constexpr int n = 1 << log_n;
  constexpr int nthreads = min_const(n, MAX_BLOCK_SIZE);
  constexpr int smem_limit = min_const(SMEM_PER_MP / min_blocks_per_mp, MAX_SMEM_PER_BLOCK);
  constexpr int smem_per_thread = min_const(max_smem_per_thread, items_per_thread,
                                            smem_limit / (nthreads * sizeof(scalar_t)));
  constexpr int mult_per_warp = n / nthreads;
  scalar_t input_val[mult_per_warp][items_per_thread];
  // const int input_idx_1 = (threadIdx.x % warpSize) + mult_per_warp * warpSize * (threadIdx.x / warpSize);
  const int input_idx_1 = (threadIdx.x & ((1 << 5) - 1)) + mult_per_warp * warpSize * (threadIdx.x >> 5);
  input_reader.load<items_per_thread, mult_per_warp>(input_val, input_idx_1);
  if (log_n <= 5) {
    b_untied_forward<min_const(log_n, 5), increasing_stride, items_per_thread>
      (twiddle_a, input_val, 0, input_idx_1);
  } else {
    __shared__ scalar_t temp_storage[nthreads * smem_per_thread];
    // constexpr int nsteps_1 = div_up_const(log_n, 2);
    constexpr int nsteps_1 = log_n <= 10 ? 5 : log_n - 5;
    constexpr int nsteps_2 = max_const(log_n - nsteps_1, 1);  // Take max to avoid compiler's warning
    constexpr int log_nwarps = min_const(max_const(log_n - 5, 1), 5);  // Take max to avoid compiler's warning
    const int input_idx_2 = ((threadIdx.x & ((1 << log_nwarps) - 1)) << nsteps_1) + (threadIdx.x >> log_nwarps);
    const int thread_idx_2 = (threadIdx.x & ((1 << log_nwarps) - 1)) * warpSize + (threadIdx.x >> log_nwarps);
    if (increasing_stride) {
      b_untied_forward<nsteps_1, true, items_per_thread, mult_per_warp>(twiddle_a, input_val, 0, input_idx_1);
      block_exchange<items_per_thread, mult_per_warp, smem_per_thread>(temp_storage, input_val, threadIdx.x, thread_idx_2, nthreads);
      b_untied_forward<nsteps_2, true, items_per_thread, mult_per_warp>(twiddle_a, input_val, nsteps_1, input_idx_2);
      // Don't need __syncthreads() before block_exchange because threads are writing to the same indices.
      block_exchange<items_per_thread, mult_per_warp, smem_per_thread>(temp_storage, input_val, thread_idx_2, threadIdx.x, nthreads);
    } else {
      block_exchange<items_per_thread, mult_per_warp, smem_per_thread>(temp_storage, input_val, threadIdx.x, thread_idx_2, nthreads);
      b_untied_forward<nsteps_2, false, items_per_thread, mult_per_warp>(twiddle_a, input_val, 0, input_idx_2);
      block_exchange<items_per_thread, mult_per_warp, smem_per_thread>(temp_storage, input_val, thread_idx_2, threadIdx.x, nthreads);
      b_untied_forward<nsteps_1, false, items_per_thread, mult_per_warp>(twiddle_a, input_val, nsteps_2, input_idx_1);
    }
  }
  output_writer.save<items_per_thread, mult_per_warp>(input_val, input_idx_1);
}

void butterfly_multiply_untied_forward_fast_cuda(const at::Tensor &twiddle,
                                                 const at::Tensor &input,
                                                 at::Tensor &output,
                                                 bool increasing_stride) {
  int batch_size = input.size(0);
  const int nstack = input.size(1);
  const int n = input.size(2);
  const int log_n = int(log2((double) n));
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "butterfly_multiply_untied_forward_fast_cuda", [&] {
    const auto twiddle_a = twiddle.packed_accessor<scalar_t, 4, at::RestrictPtrTraits, int32_t>();
    const InputReader<scalar_t> input_reader(input);
    OutputWriter<scalar_t> output_writer(output);
    dim3 block(min(n, MAX_BLOCK_SIZE));
    dim3 grid(div_up(batch_size, ITEMS_PER_THREAD_FORWARD[log_n - 1]), 1, nstack);
    auto stream = at::cuda::getCurrentCUDAStream();
    switch (log_n) {
      #define CASE_LOG_N(log_n_val) case log_n_val:                                           \
      increasing_stride ? butterfly_multiply_untied_forward_fast_cuda_kernel<log_n_val, true> \
        <<<grid, block, 0, stream>>>(twiddle_a, input_reader, output_writer, batch_size)      \
        : butterfly_multiply_untied_forward_fast_cuda_kernel<log_n_val, false>                \
        <<<grid, block, 0, stream>>>(twiddle_a, input_reader, output_writer, batch_size); break;

      MAP(CASE_LOG_N, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)
    }
  });
  // Have to keep this #undef outside the AT_DISPATCH_FLOATING_TYPES macro for it to work
  #undef CASE_LOG_N
  TORCH_CHECK(cudaGetLastError() == cudaSuccess,
     "butterfly_multiply_untied_forward_fast_cuda failed with error code ",
     cudaGetLastError());
}

template <int nsteps, bool increasing_stride, int items_per_thread,
            int mult_per_warp=1, int reg_storage_per_thread=items_per_thread,
            typename scalar_t, typename accscalar_t=at::acc_type<scalar_t, true>>
__device__ __forceinline__ void b_untied_forward_backward(const CudaAcsr<scalar_t, 4> twiddle_a,
                                                          CudaAcsr<scalar_t, 4> d_twiddle_a,
                                                          scalar_t input_val[mult_per_warp][items_per_thread],
                                                          scalar_t grad_val[mult_per_warp][items_per_thread],
                                                          int twiddle_idx_start,
                                                          int input_idx) {
  constexpr int nslices = div_up_const(items_per_thread, reg_storage_per_thread);
  static_assert(nslices == 1, "nslices not 1");
  const int s = blockIdx.y + gridDim.y * blockIdx.z;  // For conv2d butterfly as well
  scalar_t twiddle_val[nsteps][mult_per_warp][2];
  accscalar_t d_twiddle_val[nsteps][mult_per_warp][2] = {0};
  scalar_t input_val_storage[nsteps][mult_per_warp][reg_storage_per_thread];
  // Strange bug: if I use the for loop with #pragma unroll (even though nslices=1)
  // the result is wrong for n = 4096, batch_size >= 8, increasing_stride=False,
  // items_per_thread=8 (not 1, 2, or 4).
  // For now I'm disabling slicing (i.e. reg_storage_per_thread=items_per_thread always).
  // #pragma unroll
  // for (int slice = 0; slice < nslices; slice++) {
  {
    constexpr int slice = 0;
    assert(slice == 0);
    #pragma unroll
    for (int mult = 0; mult < mult_per_warp; mult++) {
      #pragma unroll
      for (int item = 0; (item < reg_storage_per_thread) && (slice * reg_storage_per_thread + item < items_per_thread); item++) {
        input_val_storage[0][mult][item] = input_val[mult][slice * reg_storage_per_thread + item];
      }
    }
    #pragma unroll
    for (int step = 0; step < nsteps; step++) {
      int log_stride = increasing_stride ? step : nsteps - 1 - step;
      int lane_mask = 1 << log_stride;
      int twiddle_idx = step + twiddle_idx_start;
      #pragma unroll
      for (int mult = 0; mult < mult_per_warp; mult++) {
        if (slice == 0) {
          twiddle_val[step][mult][0] = twiddle_a[s][twiddle_idx][0][mult * warpSize + input_idx];
          twiddle_val[step][mult][1] = twiddle_a[s][twiddle_idx][1][mult * warpSize + input_idx];
        }
        if (step < nsteps - 1) {  // Don't need input for the last step
          #pragma unroll
          for (int item = 0; (item < reg_storage_per_thread) && (slice * reg_storage_per_thread + item < items_per_thread); item++) {
            scalar_t input_val_other = log_stride < 5 ?
                __shfl_xor_sync(FULL_MASK, input_val_storage[step][mult][item], lane_mask)
              : input_val_storage[step][mult ^ (1 << (log_stride - 5))][item];
            input_val_storage[step + 1][mult][item] = twiddle_val[step][mult][0] * input_val_storage[step][mult][item]
              + twiddle_val[step][mult][1] * input_val_other;
          }
        }
      }
    }
    #pragma unroll
    for (int step = nsteps - 1; step >= 0; step--) {
      int log_stride = increasing_stride ? step : nsteps - 1 - step;
      int twiddle_idx = step + twiddle_idx_start;
      int lane_mask = 1 << log_stride;
      #pragma unroll
      for (int mult = 0; mult < mult_per_warp; mult++) {
        #pragma unroll
        for (int item = 0; (item < reg_storage_per_thread) && (slice * reg_storage_per_thread + item < items_per_thread); item++) {
          int item_offset = slice * reg_storage_per_thread + item;
          d_twiddle_val[step][mult][0] += grad_val[mult][item_offset] * input_val_storage[step][mult][item];
          scalar_t input_val_other = log_stride < 5 ?
              __shfl_xor_sync(FULL_MASK, input_val_storage[step][mult][item], lane_mask)
            : input_val_storage[step][mult ^ (1 << (log_stride - 5))][item];
          d_twiddle_val[step][mult][1] += grad_val[mult][item_offset] * input_val_other;
          if (log_stride < 5) {
            grad_val[mult][item_offset] = twiddle_val[step][mult][0] * grad_val[mult][item_offset]
              + __shfl_xor_sync(FULL_MASK, twiddle_val[step][mult][1] * grad_val[mult][item_offset], lane_mask);
          }
        }
        if (slice == nslices - 1) {
          atomicAdd(&d_twiddle_a[s][twiddle_idx][0][mult * warpSize + input_idx], d_twiddle_val[step][mult][0]);
          atomicAdd(&d_twiddle_a[s][twiddle_idx][1][mult * warpSize + input_idx], d_twiddle_val[step][mult][1]);
        }
      }
      if (log_stride >= 5) {
        int mult_stride = 1 << (log_stride - 5);
        #pragma unroll
        for (int m = 0; m < mult_per_warp / 2; m++) {
          int low_order_bits = m & (mult_stride - 1);  // int low_order_bits = m % mult_stride;
          int mult = 2 * (m - low_order_bits) + low_order_bits;
          #pragma unroll
          for (int item = 0; (item < reg_storage_per_thread) && (slice * reg_storage_per_thread + item < items_per_thread); item++) {
            int item_offset = slice * reg_storage_per_thread + item;
            scalar_t grads[2] = {grad_val[mult][item_offset], grad_val[mult + mult_stride][item_offset]};
            // The order of twiddle[1] is swapped by design
            grad_val[mult][item_offset] = twiddle_val[step][mult][0] * grads[0]
              + twiddle_val[step][mult + mult_stride][1] * grads[1];
            grad_val[mult + mult_stride][item_offset] = twiddle_val[step][mult][1] * grads[0]
              + twiddle_val[step][mult + mult_stride][0] * grads[1];
          }
        }
      }
    }
  }
}

template <int log_n, bool increasing_stride,
            int items_per_thread=ITEMS_PER_THREAD_BACKWARD[log_n - 1],
            int max_reg_storage_per_thread=items_per_thread,
            int min_blocks_per_mp=MIN_BLOCKS_PER_MP_BACKWARD[log_n - 1],
            int max_smem_per_thread=items_per_thread, typename scalar_t>
// C10_LAUNCH_BOUNDS_2 supposedly takes min(1 << log_n, 1024)
// https://github.com/pytorch/pytorch/blob/v1.1.0/c10/macros/Macros.h
// However, it doesn't seem to work correctly so I have to take min explicitly.
C10_LAUNCH_BOUNDS_2(MIN_MACRO(1 << log_n, MAX_BLOCK_SIZE), min_blocks_per_mp)
__global__ void butterfly_multiply_untied_forward_backward_fast_cuda_kernel(const CudaAcsr<scalar_t, 4> twiddle_a,
                                                                            InputReader<scalar_t> input_reader,
                                                                            InputReader<scalar_t> grad_reader,
                                                                            CudaAcsr<scalar_t, 4> d_twiddle_a,
                                                                            OutputWriter<scalar_t> d_input_writer,
                                                                            int batch_size) {
  constexpr int n = 1 << log_n;
  constexpr int nthreads = min_const(n, MAX_BLOCK_SIZE);
  constexpr int smem_limit = min_const(SMEM_PER_MP / min_blocks_per_mp, MAX_SMEM_PER_BLOCK);
  constexpr int smem_per_thread = min_const(max_smem_per_thread, items_per_thread,
                                            smem_limit / (nthreads * sizeof(scalar_t)));
  constexpr int reg_storage_per_thread = min_const(max_reg_storage_per_thread, items_per_thread);
  constexpr int mult_per_warp = n / nthreads;
  scalar_t input_val[mult_per_warp][items_per_thread];
  scalar_t grad_val[mult_per_warp][items_per_thread];
  // const int input_idx_1 = (threadIdx.x % warpSize) + mult_per_warp * warpSize * (threadIdx.x / warpSize);
  const int input_idx_1 = (threadIdx.x & ((1 << 5) - 1)) + mult_per_warp * warpSize * (threadIdx.x >> 5);
  input_reader.load<items_per_thread, mult_per_warp>(input_val, input_idx_1);
  if (log_n <= 5) {
    grad_reader.load<items_per_thread, mult_per_warp>(grad_val, input_idx_1);
    b_untied_forward_backward<min_const(log_n, 5), increasing_stride, items_per_thread, mult_per_warp, reg_storage_per_thread>
      (twiddle_a, d_twiddle_a, input_val, grad_val, 0, input_idx_1);
  } else {
    __shared__ scalar_t temp_storage[nthreads * smem_per_thread];
    // constexpr int nsteps_1 = div_up_const(log_n, 2);
    constexpr int nsteps_1 = log_n <= 10 ? 5 : log_n - 5;
    constexpr int nsteps_2 = max_const(log_n - nsteps_1, 1);  // Take max to avoid compiler's warning
    constexpr int log_nwarps = min_const(max_const(log_n - 5, 1), 5);  // Take max to avoid compiler's warning
    const int input_idx_2 = ((threadIdx.x & ((1 << log_nwarps) - 1)) << nsteps_1) + (threadIdx.x >> log_nwarps);
    const int thread_idx_2 = (threadIdx.x & ((1 << log_nwarps) - 1)) * warpSize + (threadIdx.x >> log_nwarps);
    if (increasing_stride) {
      b_untied_forward<nsteps_1, true, items_per_thread, mult_per_warp>(twiddle_a, input_val, 0, input_idx_1);
      block_exchange<items_per_thread, mult_per_warp, smem_per_thread>(temp_storage, input_val, threadIdx.x, thread_idx_2, nthreads);
      grad_reader.load<items_per_thread, mult_per_warp>(grad_val, input_idx_2);
      b_untied_forward_backward<nsteps_2, true, items_per_thread, mult_per_warp, reg_storage_per_thread>
        (twiddle_a, d_twiddle_a, input_val, grad_val, nsteps_1, input_idx_2);
      // Don't need __syncthreads() before block_exchange because threads are writing to the same indices.
      block_exchange<items_per_thread, mult_per_warp, smem_per_thread>(temp_storage, grad_val, thread_idx_2, threadIdx.x, nthreads);
      input_reader.load<items_per_thread, mult_per_warp>(input_val, input_idx_1);
      b_untied_forward_backward<nsteps_1, true, items_per_thread, mult_per_warp, reg_storage_per_thread>
        (twiddle_a, d_twiddle_a, input_val, grad_val, 0, input_idx_1);
    } else {
      block_exchange<items_per_thread, mult_per_warp, smem_per_thread>(temp_storage, input_val, threadIdx.x, thread_idx_2, nthreads);
      b_untied_forward<nsteps_2, false, items_per_thread, mult_per_warp>(twiddle_a, input_val, 0, input_idx_2);
      block_exchange<items_per_thread, mult_per_warp, smem_per_thread>(temp_storage, input_val, thread_idx_2, threadIdx.x, nthreads);
      grad_reader.load<items_per_thread, mult_per_warp>(grad_val, input_idx_1);
      b_untied_forward_backward<nsteps_1, false, items_per_thread, mult_per_warp, reg_storage_per_thread>
        (twiddle_a, d_twiddle_a, input_val, grad_val, nsteps_2, input_idx_1);
      block_exchange<items_per_thread, mult_per_warp, smem_per_thread>(temp_storage, grad_val, threadIdx.x, thread_idx_2, nthreads);
      input_reader.load<items_per_thread, mult_per_warp>(input_val, input_idx_2);
      b_untied_forward_backward<nsteps_2, false, items_per_thread, mult_per_warp, reg_storage_per_thread>
        (twiddle_a, d_twiddle_a, input_val, grad_val, 0, input_idx_2);
      block_exchange<items_per_thread, mult_per_warp, smem_per_thread>(temp_storage, grad_val, thread_idx_2, threadIdx.x, nthreads);
    }
  }
  d_input_writer.save<items_per_thread, mult_per_warp>(grad_val, input_idx_1);
}

void butterfly_multiply_untied_forward_backward_fast_cuda(const at::Tensor &twiddle,
                                                          const at::Tensor &input,
                                                          const at::Tensor &grad,
                                                          at::Tensor& d_twiddle,
                                                          at::Tensor& d_input,
                                                          bool increasing_stride) {
  int batch_size = input.size(0);
  const int nstack = input.size(1);
  const int n = input.size(2);
  const int log_n = int(log2((double) n));
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "butterfly_multiply_untied_forward_backward_fast_cuda", [&] {
    const auto twiddle_a = twiddle.packed_accessor<scalar_t, 4, at::RestrictPtrTraits, int32_t>();
    const InputReader<scalar_t> input_reader(input);
    const InputReader<scalar_t> grad_reader(grad);
    auto d_twiddle_a = d_twiddle.packed_accessor<scalar_t, 4, at::RestrictPtrTraits, int32_t>();
    OutputWriter<scalar_t> d_input_writer(d_input);
    dim3 block(min(n, MAX_BLOCK_SIZE));
    dim3 grid(div_up(batch_size, ITEMS_PER_THREAD_BACKWARD[log_n - 1]), 1, nstack);
    auto stream = at::cuda::getCurrentCUDAStream();
    switch (log_n) {
      #define CASE_LOG_N(log_n_val) case log_n_val:                                                                 \
      increasing_stride ? butterfly_multiply_untied_forward_backward_fast_cuda_kernel<log_n_val, true>              \
        <<<grid, block, 0, stream>>>(twiddle_a, input_reader, grad_reader, d_twiddle_a, d_input_writer, batch_size) \
        : butterfly_multiply_untied_forward_backward_fast_cuda_kernel<log_n_val, false>                             \
        <<<grid, block, 0, stream>>>(twiddle_a, input_reader, grad_reader, d_twiddle_a, d_input_writer, batch_size); break;

      MAP(CASE_LOG_N, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
    }
  });
  // Have to keep this #undef outside the AT_DISPATCH_FLOATING_TYPES macro for it to work
  #undef CASE_LOG_N
  TORCH_CHECK(cudaGetLastError() == cudaSuccess,
     "butterfly_multiply_untied_forward_backward_fast_cuda failed with error code ",
     cudaGetLastError());
}

template <int log_n, int items_per_thread=ITEMS_PER_THREAD_FORWARD[log_n - 1],
            int min_blocks_per_mp=MIN_BLOCKS_PER_MP_FORWARD[log_n - 1],
            int max_smem_per_thread=items_per_thread, typename scalar_t>
// C10_LAUNCH_BOUNDS_2 supposedly takes min(1 << log_n, 1024)
// https://github.com/pytorch/pytorch/blob/v1.1.0/c10/macros/Macros.h
// However, it doesn't seem to work correctly so I have to take min explicitly.
C10_LAUNCH_BOUNDS_2(MIN_MACRO(1 << log_n, MAX_BLOCK_SIZE), min_blocks_per_mp)
__global__ void butterfly_bbs_multiply_untied_forward_fast_cuda_kernel(const CudaAcsr<scalar_t, 4> twiddle_a,
                                                                       InputReader<scalar_t> input_reader,
                                                                       OutputWriter<scalar_t> output_writer,
                                                                       int batch_size,
                                                                       int nblocks) {
  constexpr int n = 1 << log_n;
  constexpr int nthreads = min_const(n, MAX_BLOCK_SIZE);
  constexpr int smem_limit = min_const(SMEM_PER_MP / min_blocks_per_mp, MAX_SMEM_PER_BLOCK);
  constexpr int smem_per_thread = min_const(max_smem_per_thread, items_per_thread,
                                            smem_limit / (nthreads * sizeof(scalar_t)));
  constexpr int mult_per_warp = n / nthreads;
  scalar_t input_val[mult_per_warp][items_per_thread];
  // const int input_idx_1 = (threadIdx.x % warpSize) + mult_per_warp * warpSize * (threadIdx.x / warpSize);
  const int input_idx_1 = (threadIdx.x & ((1 << 5) - 1)) + mult_per_warp * warpSize * (threadIdx.x >> 5);
  input_reader.load<items_per_thread, mult_per_warp>(input_val, input_idx_1);
  if (log_n <= 5) {
    for (int block = 0; block < nblocks; block++) {
      b_untied_forward<min_const(log_n, 5), false, items_per_thread>
        (twiddle_a, input_val, block * 2 * log_n, input_idx_1);
      b_untied_forward<min_const(log_n, 5), true, items_per_thread>
        (twiddle_a, input_val, (block * 2 + 1) * log_n, input_idx_1);
    }
  } else {
    __shared__ scalar_t temp_storage[nthreads * smem_per_thread];
    constexpr int nsteps_1 = log_n <= 10 ? 5 : log_n - 5;
    constexpr int nsteps_2 = max_const(log_n - nsteps_1, 1);  // Take max to avoid compiler's warning
    constexpr int log_nwarps = min_const(max_const(log_n - 5, 1), 5);  // Take max to avoid compiler's warning
    const int input_idx_2 = ((threadIdx.x & ((1 << log_nwarps) - 1)) << nsteps_1) + (threadIdx.x >> log_nwarps);
    const int thread_idx_2 = (threadIdx.x & ((1 << log_nwarps) - 1)) * warpSize + (threadIdx.x >> log_nwarps);
    block_exchange<items_per_thread, mult_per_warp, smem_per_thread>(temp_storage, input_val, threadIdx.x, thread_idx_2, nthreads);
    for (int block = 0; block < nblocks; block++) {
      b_untied_forward<nsteps_2, false, items_per_thread, mult_per_warp>
        (twiddle_a, input_val, block * 2 * log_n, input_idx_2);
      block_exchange<items_per_thread, mult_per_warp, smem_per_thread>(temp_storage, input_val, thread_idx_2, threadIdx.x, nthreads);
      b_untied_forward<nsteps_1, false, items_per_thread, mult_per_warp>
        (twiddle_a, input_val, block * 2 * log_n + nsteps_2, input_idx_1);
      b_untied_forward<nsteps_1, true, items_per_thread, mult_per_warp>
        (twiddle_a, input_val, (block * 2 + 1) * log_n, input_idx_1);
      block_exchange<items_per_thread, mult_per_warp, smem_per_thread>(temp_storage, input_val, threadIdx.x, thread_idx_2, nthreads);
      b_untied_forward<nsteps_2, true, items_per_thread, mult_per_warp>
        (twiddle_a, input_val, (block * 2 + 1) * log_n + nsteps_1, input_idx_2);
    }
    block_exchange<items_per_thread, mult_per_warp, smem_per_thread>(temp_storage, input_val, thread_idx_2, threadIdx.x, nthreads);
  }
  output_writer.save<items_per_thread, mult_per_warp>(input_val, input_idx_1);
}

void butterfly_bbs_multiply_untied_forward_fast_cuda(const at::Tensor &twiddle,
                                                     const at::Tensor &input,
                                                     at::Tensor &output) {
  int batch_size = input.size(0);
  const int nstack = input.size(1);
  const int n = input.size(2);
  const int log_n = int(log2((double) n));
  const int nblocks = twiddle.size(1) / (2 * log_n);
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "butterfly_bbs_multiply_untied_forward_fast_cuda", [&] {
    const auto twiddle_a = twiddle.packed_accessor<scalar_t, 4, at::RestrictPtrTraits, int32_t>();
    const InputReader<scalar_t> input_reader(input);
    OutputWriter<scalar_t> output_writer(output);
    dim3 block(min(n, MAX_BLOCK_SIZE));
    dim3 grid(div_up(batch_size, ITEMS_PER_THREAD_FORWARD[log_n - 1]), 1, nstack);
    auto stream = at::cuda::getCurrentCUDAStream();
    switch (log_n) {
      #define CASE_LOG_N(log_n_val) case log_n_val:                     \
      butterfly_bbs_multiply_untied_forward_fast_cuda_kernel<log_n_val> \
        <<<grid, block, 0, stream>>>(twiddle_a, input_reader, output_writer, batch_size, nblocks); break;
      MAP(CASE_LOG_N, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)
    }
  });
  // Have to keep this #undef outside the AT_DISPATCH_FLOATING_TYPES macro for it to work
  #undef CASE_LOG_N
  TORCH_CHECK(cudaGetLastError() == cudaSuccess,
     "butterfly_bbs_multiply_untied_forward_fast_cuda failed with error code ",
     cudaGetLastError());
}

template <int log_n, int items_per_thread=ITEMS_PER_THREAD_BACKWARD[log_n - 1],
            int max_reg_storage_per_thread=items_per_thread,
            int min_blocks_per_mp=MIN_BLOCKS_PER_MP_BACKWARD[log_n - 1],
            int max_smem_per_thread=items_per_thread, typename scalar_t>
C10_LAUNCH_BOUNDS_2(MIN_MACRO(1 << log_n, MAX_BLOCK_SIZE), min_blocks_per_mp)
__global__ void butterfly_bbs_multiply_untied_forward_backward_fast_cuda_kernel(const CudaAcsr<scalar_t, 4> twiddle_a,
                                                                                InputReader<scalar_t> input_reader,
                                                                                InputReader<scalar_t> grad_reader,
                                                                                IntermediateStorage<scalar_t> inter_storage,
                                                                                CudaAcsr<scalar_t, 4> d_twiddle_a,
                                                                                OutputWriter<scalar_t> d_input_writer,
                                                                                int batch_size,
                                                                                int nblocks) {
  constexpr int n = 1 << log_n;
  constexpr int nthreads = min_const(n, MAX_BLOCK_SIZE);
  constexpr int smem_limit = min_const(SMEM_PER_MP / min_blocks_per_mp, MAX_SMEM_PER_BLOCK);
  constexpr int smem_per_thread = min_const(max_smem_per_thread, items_per_thread,
                                            smem_limit / (nthreads * sizeof(scalar_t)));
  constexpr int reg_storage_per_thread = min_const(max_reg_storage_per_thread, items_per_thread);
  constexpr int mult_per_warp = n / nthreads;
  scalar_t input_val[mult_per_warp][items_per_thread];
  scalar_t grad_val[mult_per_warp][items_per_thread];
  // const int input_idx_1 = (threadIdx.x % warpSize) + mult_per_warp * warpSize * (threadIdx.x / warpSize);
  const int input_idx_1 = (threadIdx.x & ((1 << 5) - 1)) + mult_per_warp * warpSize * (threadIdx.x >> 5);
  input_reader.load<items_per_thread, mult_per_warp>(input_val, input_idx_1);
  if (log_n <= 5) {
    for (int block = 0; block < nblocks; block++) {
      b_untied_forward<min_const(log_n, 5), false, items_per_thread>
        (twiddle_a, input_val, block * 2 * log_n, input_idx_1);
      if (block < nblocks - 1) {
        inter_storage.save<items_per_thread>(input_val, input_idx_1, block * 2);
        b_untied_forward<min_const(log_n, 5), true, items_per_thread>
          (twiddle_a, input_val, (block * 2 + 1) * log_n, input_idx_1);
        inter_storage.save<items_per_thread>(input_val, input_idx_1, block * 2 + 1);
      }
    }
    grad_reader.load<items_per_thread, mult_per_warp>(grad_val, input_idx_1);
    for (int block = nblocks - 1; block >= 0; block--) {
      if (block < nblocks - 1) {
        inter_storage.load<items_per_thread>(input_val, input_idx_1, block * 2);
      }
      b_untied_forward_backward<min_const(log_n, 5), true, items_per_thread, mult_per_warp, reg_storage_per_thread>
        (twiddle_a, d_twiddle_a, input_val, grad_val, (block * 2 + 1) * log_n, input_idx_1);
      block == 0 ? input_reader.load<items_per_thread>(input_val, input_idx_1)
        : inter_storage.load<items_per_thread>(input_val, input_idx_1, (block - 1) * 2 + 1);
      b_untied_forward_backward<min_const(log_n, 5), false, items_per_thread, mult_per_warp, reg_storage_per_thread>
        (twiddle_a, d_twiddle_a, input_val, grad_val, block * 2 * log_n, input_idx_1);
    }
  } else {
    __shared__ scalar_t temp_storage[nthreads * smem_per_thread];
    // constexpr int nsteps_1 = div_up_const(log_n, 2);
    constexpr int nsteps_1 = log_n <= 10 ? 5 : log_n - 5;
    constexpr int nsteps_2 = max_const(log_n - nsteps_1, 1);  // Take max to avoid compiler's warning
    constexpr int log_nwarps = min_const(max_const(log_n - 5, 1), 5);  // Take max to avoid compiler's warning
    const int input_idx_2 = ((threadIdx.x & ((1 << log_nwarps) - 1)) << nsteps_1) + (threadIdx.x >> log_nwarps);
    const int thread_idx_2 = (threadIdx.x & ((1 << log_nwarps) - 1)) * warpSize + (threadIdx.x >> log_nwarps);
    block_exchange<items_per_thread, mult_per_warp, smem_per_thread>(temp_storage, input_val, threadIdx.x, thread_idx_2, nthreads);
    for (int block = 0; block < nblocks; block++) {
      b_untied_forward<nsteps_2, false, items_per_thread, mult_per_warp>
        (twiddle_a, input_val, block * 2 * log_n, input_idx_2);
      block_exchange<items_per_thread, mult_per_warp, smem_per_thread>(temp_storage, input_val, thread_idx_2, threadIdx.x, nthreads);
      inter_storage.save<items_per_thread, mult_per_warp>(input_val, input_idx_1, block * 4);
      b_untied_forward<nsteps_1, false, items_per_thread, mult_per_warp>
        (twiddle_a, input_val, block * 2 * log_n + nsteps_2, input_idx_1);
      inter_storage.save<items_per_thread, mult_per_warp>(input_val, input_idx_1, block * 4 + 1);
      b_untied_forward<nsteps_1, true, items_per_thread, mult_per_warp>
        (twiddle_a, input_val, (block * 2 + 1) * log_n, input_idx_1);
      block_exchange<items_per_thread, mult_per_warp, smem_per_thread>(temp_storage, input_val, threadIdx.x, thread_idx_2, nthreads);
      if (block < nblocks - 1) {
        // We can store using input_idx_1 instead of input_idx_2 since we'll load using input_idx_1 as well
        inter_storage.save<items_per_thread, mult_per_warp>(input_val, input_idx_1, block * 4 + 2);
        b_untied_forward<nsteps_2, true, items_per_thread, mult_per_warp>
          (twiddle_a, input_val, (block * 2 + 1) * log_n + nsteps_1, input_idx_2);
        inter_storage.save<items_per_thread, mult_per_warp>(input_val, input_idx_1, block * 4 + 3);
      }
    }
    grad_reader.load<items_per_thread, mult_per_warp>(grad_val, input_idx_2);
    for (int block = nblocks - 1; block >= 0; block--) {
      if (block < nblocks - 1) {
        inter_storage.load<items_per_thread, mult_per_warp>(input_val, input_idx_1, block * 4 + 2);
      }
      b_untied_forward_backward<nsteps_2, true, items_per_thread, mult_per_warp, reg_storage_per_thread>
        (twiddle_a, d_twiddle_a, input_val, grad_val, (block * 2 + 1) * log_n + nsteps_1, input_idx_2);
      block_exchange<items_per_thread, mult_per_warp, smem_per_thread>(temp_storage, grad_val, thread_idx_2, threadIdx.x, nthreads);
      inter_storage.load<items_per_thread, mult_per_warp>(input_val, input_idx_1, block * 4 + 1);
      b_untied_forward_backward<nsteps_1, true, items_per_thread, mult_per_warp, reg_storage_per_thread>
        (twiddle_a, d_twiddle_a, input_val, grad_val, (block * 2 + 1) * log_n, input_idx_1);
      inter_storage.load<items_per_thread, mult_per_warp>(input_val, input_idx_1, block * 4);
      b_untied_forward_backward<nsteps_1, false, items_per_thread, mult_per_warp>
        (twiddle_a, d_twiddle_a, input_val, grad_val, block * 2 * log_n + nsteps_2, input_idx_1);
      block_exchange<items_per_thread, mult_per_warp, smem_per_thread>(temp_storage, grad_val, threadIdx.x, thread_idx_2, nthreads);
      block == 0 ? input_reader.load<items_per_thread, mult_per_warp>(input_val, input_idx_2)
        : inter_storage.load<items_per_thread, mult_per_warp>(input_val, input_idx_1, (block - 1) * 4 + 3);
      b_untied_forward_backward<nsteps_2, false, items_per_thread, mult_per_warp>
        (twiddle_a, d_twiddle_a, input_val, grad_val, block * 2 * log_n, input_idx_2);
    }
    block_exchange<items_per_thread, mult_per_warp, smem_per_thread>(temp_storage, grad_val, thread_idx_2, threadIdx.x, nthreads);
  }
  d_input_writer.save<items_per_thread, mult_per_warp>(grad_val, input_idx_1);
}

void butterfly_bbs_multiply_untied_forward_backward_fast_cuda(const at::Tensor &twiddle,
                                                              const at::Tensor &input,
                                                              const at::Tensor &grad,
                                                              at::Tensor& d_twiddle,
                                                              at::Tensor& d_input) {
  int batch_size = input.size(0);
  const int nstack = input.size(1);
  const int n = input.size(2);
  const int log_n = int(log2((double) n));
  const int nblocks = twiddle.size(1) / (2 * log_n);
  auto intermediate_storage = at::empty({log_n <= 5 ? (nblocks - 1) * 2 : (nblocks - 1) * 4 + 2, batch_size, nstack, n},
                                        at::dtype(input.dtype()).device(input.device()));
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "butterfly_bbs_multiply_untied_forward_backward_fast_cuda", [&] {
    const auto twiddle_a = twiddle.packed_accessor<scalar_t, 4, at::RestrictPtrTraits, int32_t>();
    const InputReader<scalar_t> input_reader(input);
    const InputReader<scalar_t> grad_reader(grad);
    IntermediateStorage<scalar_t> inter_storage(intermediate_storage);
    auto d_twiddle_a = d_twiddle.packed_accessor<scalar_t, 4, at::RestrictPtrTraits, int32_t>();
    OutputWriter<scalar_t> d_input_writer(d_input);
    dim3 block(min(n, MAX_BLOCK_SIZE));
    dim3 grid(div_up(batch_size, ITEMS_PER_THREAD_BACKWARD[log_n - 1]), 1, nstack);
    auto stream = at::cuda::getCurrentCUDAStream();
    switch (log_n) {
      #define CASE_LOG_N(log_n_val) case log_n_val:                              \
      butterfly_bbs_multiply_untied_forward_backward_fast_cuda_kernel<log_n_val> \
        <<<grid, block, 0, stream>>>(twiddle_a, input_reader, grad_reader, inter_storage, d_twiddle_a, d_input_writer, batch_size, nblocks); break;
      MAP(CASE_LOG_N, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
    }
  });
  // Have to keep this #undef outside the AT_DISPATCH_FLOATING_TYPES macro for it to work
  #undef CASE_LOG_N
  TORCH_CHECK(cudaGetLastError() == cudaSuccess,
     "butterfly_bbs_multiply_untied_forward_backward_fast_cuda failed with error code ",
     cudaGetLastError());
}

template <int nsteps, bool increasing_stride, int items_per_thread,
            int mult_per_warp=1, typename scalar_t>
__device__ __forceinline__ void b_ortho_untied_forward(const CudaAcsr<scalar_t, 3> twiddle_cos_a,
                                                       const CudaAcsr<scalar_t, 3> twiddle_sin_a,
                                                       scalar_t input_val[mult_per_warp][items_per_thread],
                                                       int twiddle_idx_start,
                                                       int input_idx,
                                                       int log_input_stride_start) {
  const int s = blockIdx.y + gridDim.y * blockIdx.z;  // For conv2d butterfly_ortho as well
  #pragma unroll
  // TODO: for loop over mult first instead of step first,
  // will have to split into 2 parts: intra-thread and intra-warp.
  for (int step = 0; step < nsteps; step++) {
    int log_stride = increasing_stride ? step : nsteps - 1 - step;
    int log_input_stride = log_input_stride_start + log_stride;
    int twiddle_idx = twiddle_idx_start + step;
    if (log_stride < 5) {
      int lane_mask = 1 << log_stride;
      #pragma unroll
      for (int mult = 0; mult < mult_per_warp; mult++) {
        // TODO: make num thread per warp an input argument
        int idx = mult * warpSize + input_idx;
        int low_order_bits = idx & ((1 << log_input_stride) - 1);  // int low_order_bits = idx % (1 << log_input_stride);
        // Bit manipulation to delete the bit at log_input_stride
        int index_access = ((idx >> (log_input_stride + 1)) << log_input_stride) + low_order_bits;
        bool odd = (idx >> log_input_stride) & 1U;
        scalar_t twiddle_val_mine = !odd ? twiddle_cos_a[s][twiddle_idx][index_access] : twiddle_sin_a[s][twiddle_idx][index_access];
        scalar_t twiddle_val_other = __shfl_xor_sync(FULL_MASK, twiddle_val_mine, lane_mask);
        const scalar_t twiddle_val[2] = {!odd ? twiddle_val_mine : twiddle_val_other,
                                         !odd ? -twiddle_val_other : twiddle_val_mine};
        // if (not odd) {
        //   twiddle_val[0] = -twiddle_cos_a[s][twiddle_idx][index_access];
        //   twiddle_val[1] = -twiddle_sin_a[s][twiddle_idx][index_access];
        // }
        // scalar_t twiddle_val_exch[2] = {__shfl_xor_sync(FULL_MASK, twiddle_val[0], lane_mask),
        //                                 __shfl_xor_sync(FULL_MASK, twiddle_val[1], lane_mask)};
        // if (odd) {
        //   twiddle_val[0] = twiddle_val_exch[0];
        //   twiddle_val[1] = -twiddle_val_exch[1];
        // }
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
        int idx = mult * warpSize + input_idx;
        low_order_bits = idx & ((1 << log_input_stride) - 1);  // int low_order_bits = idx % (1 << log_input_stride);
        // int index_access = ((idx & ~(1U << log_input_stride)) - low_order_bits) / 2 + low_order_bits;
        int index_access = ((idx >> (log_input_stride + 1)) << log_input_stride) + low_order_bits;
        const scalar_t twiddle_val[2] = {twiddle_cos_a[s][twiddle_idx][index_access],
                                         twiddle_sin_a[s][twiddle_idx][index_access]};
        #pragma unroll
        for (int item = 0; item < items_per_thread; item++) {
          scalar_t inputs[2] = {input_val[mult][item], input_val[mult + mult_stride][item]};
          input_val[mult][item] = twiddle_val[0] * inputs[0] - twiddle_val[1] * inputs[1];
          input_val[mult + mult_stride][item] = twiddle_val[1] * inputs[0] + twiddle_val[0] * inputs[1];
        }
      }
    }
  }
}

template <int log_n, bool increasing_stride,
            int items_per_thread=ITEMS_PER_THREAD_ORTHO_FORWARD[log_n - 1],
            int min_blocks_per_mp=MIN_BLOCKS_PER_MP_ORTHO_FORWARD[log_n - 1],
            int max_smem_per_thread=items_per_thread, typename scalar_t>
// C10_LAUNCH_BOUNDS_2 supposedly takes min(1 << log_n, 1024)
// https://github.com/pytorch/pytorch/blob/v1.1.0/c10/macros/Macros.h
// However, it doesn't seem to work correctly so I have to take min explicitly.
C10_LAUNCH_BOUNDS_2(MIN_MACRO(1 << log_n, MAX_BLOCK_SIZE), min_blocks_per_mp)
__global__ void butterfly_ortho_multiply_untied_forward_fast_cuda_kernel(const CudaAcsr<scalar_t, 3> twiddle_cos_a,
                                                                         const CudaAcsr<scalar_t, 3> twiddle_sin_a,
                                                                         InputReader<scalar_t> input_reader,
                                                                         OutputWriter<scalar_t> output_writer,
                                                                         int batch_size) {
  constexpr int n = 1 << log_n;
  constexpr int nthreads = min_const(n, MAX_BLOCK_SIZE);
  constexpr int smem_limit = min_const(SMEM_PER_MP / min_blocks_per_mp, MAX_SMEM_PER_BLOCK);
  constexpr int smem_per_thread = min_const(max_smem_per_thread, items_per_thread,
                                            smem_limit / (nthreads * sizeof(scalar_t)));
  constexpr int mult_per_warp = n / nthreads;
  scalar_t input_val[mult_per_warp][items_per_thread];
  // const int input_idx_1 = (threadIdx.x % warpSize) + mult_per_warp * warpSize * (threadIdx.x / warpSize);
  const int input_idx_1 = (threadIdx.x & ((1 << 5) - 1)) + mult_per_warp * warpSize * (threadIdx.x >> 5);
  input_reader.load<items_per_thread, mult_per_warp>(input_val, input_idx_1);
  if (log_n <= 5) {
    b_ortho_untied_forward<min_const(log_n, 5), increasing_stride, items_per_thread>
      (twiddle_cos_a, twiddle_sin_a, input_val, 0, input_idx_1, 0);
  } else {
    __shared__ scalar_t temp_storage[nthreads * smem_per_thread];
    // constexpr int nsteps_1 = div_up_const(log_n, 2);
    constexpr int nsteps_1 = log_n <= 10 ? 5 : log_n - 5;
    constexpr int nsteps_2 = max_const(log_n - nsteps_1, 1);  // Take max to avoid compiler's warning
    constexpr int log_nwarps = min_const(max_const(log_n - 5, 1), 5);  // Take max to avoid compiler's warning
    const int input_idx_2 = ((threadIdx.x & ((1 << log_nwarps) - 1)) << nsteps_1) + (threadIdx.x >> log_nwarps);
    const int thread_idx_2 = (threadIdx.x & ((1 << log_nwarps) - 1)) * warpSize + (threadIdx.x >> log_nwarps);
    if (increasing_stride) {
      b_ortho_untied_forward<nsteps_1, true, items_per_thread, mult_per_warp>
        (twiddle_cos_a, twiddle_sin_a, input_val, 0, input_idx_1, 0);
      block_exchange<items_per_thread, mult_per_warp, smem_per_thread>(temp_storage, input_val, threadIdx.x, thread_idx_2, nthreads);
      b_ortho_untied_forward<nsteps_2, true, items_per_thread, mult_per_warp>
        (twiddle_cos_a, twiddle_sin_a, input_val, nsteps_1, input_idx_2, nsteps_1);
      // Don't need __syncthreads() before block_exchange because threads are writing to the same indices.
      block_exchange<items_per_thread, mult_per_warp, smem_per_thread>(temp_storage, input_val, thread_idx_2, threadIdx.x, nthreads);
    } else {
      block_exchange<items_per_thread, mult_per_warp, smem_per_thread>(temp_storage, input_val, threadIdx.x, thread_idx_2, nthreads);
      b_ortho_untied_forward<nsteps_2, false, items_per_thread, mult_per_warp>
        (twiddle_cos_a, twiddle_sin_a, input_val, 0, input_idx_2, nsteps_1);
      block_exchange<items_per_thread, mult_per_warp, smem_per_thread>(temp_storage, input_val, thread_idx_2, threadIdx.x, nthreads);
      b_ortho_untied_forward<nsteps_1, false, items_per_thread, mult_per_warp>
        (twiddle_cos_a, twiddle_sin_a, input_val, nsteps_2, input_idx_1, 0);
    }
  }
  output_writer.save<items_per_thread, mult_per_warp>(input_val, input_idx_1);
}

void butterfly_ortho_multiply_untied_forward_fast_cuda(const at::Tensor &twiddle_cos,
                                                       const at::Tensor &twiddle_sin,
                                                       const at::Tensor &input,
                                                       at::Tensor &output,
                                                       bool increasing_stride) {
  int batch_size = input.size(0);
  const int nstack = input.size(1);
  const int n = input.size(2);
  const int log_n = int(log2((double) n));
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "butterfly_ortho_multiply_untied_forward_fast_cuda", [&] {
    const auto twiddle_cos_a = twiddle_cos.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>();
    const auto twiddle_sin_a = twiddle_sin.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>();
    const InputReader<scalar_t> input_reader(input);
    OutputWriter<scalar_t> output_writer(output);
    dim3 block(min(n, MAX_BLOCK_SIZE));
    dim3 grid(div_up(batch_size, ITEMS_PER_THREAD_ORTHO_FORWARD[log_n - 1]), 1, nstack);
    auto stream = at::cuda::getCurrentCUDAStream();
    switch (log_n) {
      #define CASE_LOG_N(log_n_val) case log_n_val:                                                         \
      increasing_stride ? butterfly_ortho_multiply_untied_forward_fast_cuda_kernel<log_n_val, true>         \
        <<<grid, block, 0, stream>>>(twiddle_cos_a, twiddle_sin_a, input_reader, output_writer, batch_size) \
        : butterfly_ortho_multiply_untied_forward_fast_cuda_kernel<log_n_val, false>                        \
        <<<grid, block, 0, stream>>>(twiddle_cos_a, twiddle_sin_a, input_reader, output_writer, batch_size); break;
      MAP(CASE_LOG_N, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)
    }
  });
  // Have to keep this #undef outside the AT_DISPATCH_FLOATING_TYPES macro for it to work
  #undef CASE_LOG_N
  TORCH_CHECK(cudaGetLastError() == cudaSuccess,
     "butterfly_ortho_multiply_untied_forward_fast_cuda failed with error code ",
     cudaGetLastError());
}

template <int nsteps, bool increasing_stride, int items_per_thread,
            int mult_per_warp=1, typename scalar_t,
            typename accscalar_t=at::acc_type<scalar_t, true>>
__device__ __forceinline__ void b_ortho_untied_backward(const CudaAcsr<scalar_t, 3> twiddle_cos_a,
                                                        const CudaAcsr<scalar_t, 3> twiddle_sin_a,
                                                        CudaAcsr<scalar_t, 3> d_twiddle_a,
                                                        scalar_t output_val[mult_per_warp][items_per_thread],
                                                        scalar_t grad_val[mult_per_warp][items_per_thread],
                                                        int twiddle_idx_start,
                                                        int input_idx,
                                                        int log_input_stride_start) {
  const int s = blockIdx.y + gridDim.y * blockIdx.z;  // For conv2d butterfly_ortho as well
  #pragma unroll
  for (int step = nsteps - 1; step >= 0; step--) {
    int log_stride = increasing_stride ? step : nsteps - 1 - step;
    int log_input_stride = log_input_stride_start + log_stride;
    int twiddle_idx = step + twiddle_idx_start;
    if (log_stride < 5) {
      int lane_mask = 1 << log_stride;
      #pragma unroll
      for (int mult = 0; mult < mult_per_warp; mult++) {
        int idx = mult * warpSize + input_idx;
        int low_order_bits = idx & ((1 << log_input_stride) - 1);  // int low_order_bits = idx % (1 << log_input_stride);
        // Bit manipulation to delete the bit at log_input_stride
        int index_access = ((idx >> (log_input_stride + 1)) << log_input_stride) + low_order_bits;
        bool odd = (idx >> log_input_stride) & 1U;
        scalar_t twiddle_val_mine = !odd ? twiddle_cos_a[s][twiddle_idx][index_access] : twiddle_sin_a[s][twiddle_idx][index_access];
        scalar_t twiddle_val_other = __shfl_xor_sync(FULL_MASK, twiddle_val_mine, lane_mask);
        const scalar_t twiddle_val[2] = {!odd ? twiddle_val_mine : twiddle_val_other,
                                        !odd ? twiddle_val_other : -twiddle_val_mine};
        accscalar_t d_twiddle_val = 0;
        #pragma unroll
        for (int item = 0; item < items_per_thread; item++) {
          scalar_t output_val_other = __shfl_xor_sync(FULL_MASK, output_val[mult][item], lane_mask);
          output_val[mult][item] = twiddle_val[0] * output_val[mult][item] + twiddle_val[1] * output_val_other;
          scalar_t grad_val_other = __shfl_xor_sync(FULL_MASK, grad_val[mult][item], lane_mask);
          output_val_other = __shfl_xor_sync(FULL_MASK, output_val[mult][item], lane_mask);
          if (!odd) {
            d_twiddle_val
              += (grad_val[mult][item] * output_val[mult][item] + grad_val_other * output_val_other) * (-twiddle_val[1])
              + (-grad_val[mult][item] * output_val_other + grad_val_other * output_val[mult][item]) * twiddle_val[0];
          }
          grad_val[mult][item] = twiddle_val[0] * grad_val[mult][item] + twiddle_val[1] * grad_val_other;
        }
        if (!odd) {
          atomicAdd(&d_twiddle_a[s][twiddle_idx][index_access], d_twiddle_val);
        }
      }
    } else {
      int mult_stride = 1 << (log_stride - 5);
      #pragma unroll
      for (int m = 0; m < mult_per_warp / 2; m++) {
        int low_order_bits = m & (mult_stride - 1);  // int low_order_bits = m % mult_stride;
        int mult = 2 * (m - low_order_bits) + low_order_bits;
        int idx = mult * warpSize + input_idx;
        low_order_bits = idx & ((1 << log_input_stride) - 1);  // int low_order_bits = idx % (1 << log_input_stride);
        int index_access = ((idx >> (log_input_stride + 1)) << log_input_stride) + low_order_bits;
        const scalar_t twiddle_val[2] = {twiddle_cos_a[s][twiddle_idx][index_access],
                                         twiddle_sin_a[s][twiddle_idx][index_access]};
        accscalar_t d_twiddle_val = 0;
        #pragma unroll
        for (int item = 0; item < items_per_thread; item++) {
          scalar_t outputs[2] = {output_val[mult][item], output_val[mult + mult_stride][item]};
          output_val[mult][item] = twiddle_val[0] * outputs[0] + twiddle_val[1] * outputs[1];
          output_val[mult + mult_stride][item] = -twiddle_val[1] * outputs[0] + twiddle_val[0] * outputs[1];
          scalar_t grads[2] = {grad_val[mult][item], grad_val[mult + mult_stride][item]};
          d_twiddle_val
            += (grads[0] * output_val[mult][item] + grads[1] * output_val[mult + mult_stride][item]) * (-twiddle_val[1])
            + (-grads[0] * output_val[mult + mult_stride][item] + grads[1] * output_val[mult][item]) * twiddle_val[0];
          grad_val[mult][item] = twiddle_val[0] * grads[0] + twiddle_val[1] * grads[1];
          grad_val[mult + mult_stride][item] = -twiddle_val[1] * grads[0] + twiddle_val[0] * grads[1];
        }
        atomicAdd(&d_twiddle_a[s][twiddle_idx][index_access], d_twiddle_val);
      }
    }
  }
}

template <int log_n, bool increasing_stride,
            int items_per_thread=ITEMS_PER_THREAD_ORTHO_BACKWARD[log_n - 1],
            int min_blocks_per_mp=MIN_BLOCKS_PER_MP_ORTHO_BACKWARD[log_n - 1],
            int max_smem_per_thread=items_per_thread, typename scalar_t>
// C10_LAUNCH_BOUNDS_2 supposedly takes min(1 << log_n, 1024)
// https://github.com/pytorch/pytorch/blob/v1.1.0/c10/macros/Macros.h
// However, it doesn't seem to work correctly so I have to take min explicitly.
C10_LAUNCH_BOUNDS_2(MIN_MACRO(1 << log_n, MAX_BLOCK_SIZE), min_blocks_per_mp)
__global__ void butterfly_ortho_multiply_untied_backward_fast_cuda_kernel(const CudaAcsr<scalar_t, 3> twiddle_cos_a,
                                                                          const CudaAcsr<scalar_t, 3> twiddle_sin_a,
                                                                          InputReader<scalar_t> output_reader,
                                                                          InputReader<scalar_t> grad_reader,
                                                                          CudaAcsr<scalar_t, 3> d_twiddle_a,
                                                                          OutputWriter<scalar_t> d_input_writer,
                                                                          int batch_size) {
  constexpr int n = 1 << log_n;
  constexpr int nthreads = min_const(n, MAX_BLOCK_SIZE);
  constexpr int smem_limit = min_const(SMEM_PER_MP / min_blocks_per_mp, MAX_SMEM_PER_BLOCK);
  constexpr int smem_per_thread = min_const(max_smem_per_thread, items_per_thread,
                                            smem_limit / (nthreads * sizeof(scalar_t)));
  constexpr int mult_per_warp = n / nthreads;
  scalar_t output_val[mult_per_warp][items_per_thread];
  scalar_t grad_val[mult_per_warp][items_per_thread];
  // const int input_idx_1 = (threadIdx.x % warpSize) + mult_per_warp * warpSize * (threadIdx.x / warpSize);
  const int input_idx_1 = (threadIdx.x & ((1 << 5) - 1)) + mult_per_warp * warpSize * (threadIdx.x >> 5);
  output_reader.load<items_per_thread, mult_per_warp>(output_val, input_idx_1);
  if (log_n <= 5) {
    grad_reader.load<items_per_thread, mult_per_warp>(grad_val, input_idx_1);
    b_ortho_untied_backward<min_const(log_n, 5), increasing_stride, items_per_thread, mult_per_warp>
      (twiddle_cos_a, twiddle_sin_a, d_twiddle_a, output_val, grad_val, 0, input_idx_1, 0);
  } else {
    __shared__ scalar_t temp_storage[nthreads * smem_per_thread];
    // constexpr int nsteps_1 = div_up_const(log_n, 2);
    constexpr int nsteps_1 = log_n <= 10 ? 5 : log_n - 5;
    constexpr int nsteps_2 = max_const(log_n - nsteps_1, 1);  // Take max to avoid compiler's warning
    constexpr int log_nwarps = min_const(max_const(log_n - 5, 1), 5);  // Take max to avoid compiler's warning
    const int input_idx_2 = ((threadIdx.x & ((1 << log_nwarps) - 1)) << nsteps_1) + (threadIdx.x >> log_nwarps);
    const int thread_idx_2 = (threadIdx.x & ((1 << log_nwarps) - 1)) * warpSize + (threadIdx.x >> log_nwarps);
    if (increasing_stride) {
      block_exchange<items_per_thread, mult_per_warp, smem_per_thread>(temp_storage, output_val, threadIdx.x, thread_idx_2, nthreads);
      grad_reader.load<items_per_thread, mult_per_warp>(grad_val, input_idx_2);
      b_ortho_untied_backward<nsteps_2, true, items_per_thread, mult_per_warp>
        (twiddle_cos_a, twiddle_sin_a, d_twiddle_a, output_val, grad_val, nsteps_1, input_idx_2, nsteps_1);
      block_exchange<items_per_thread, mult_per_warp, smem_per_thread>(temp_storage, grad_val, thread_idx_2, threadIdx.x, nthreads);
      __syncthreads();
      block_exchange<items_per_thread, mult_per_warp, smem_per_thread>(temp_storage, output_val, thread_idx_2, threadIdx.x, nthreads);
      b_ortho_untied_backward<nsteps_1, true, items_per_thread, mult_per_warp>
        (twiddle_cos_a, twiddle_sin_a, d_twiddle_a, output_val, grad_val, 0, input_idx_1, 0);
    } else {
      grad_reader.load<items_per_thread, mult_per_warp>(grad_val, input_idx_1);
      b_ortho_untied_backward<nsteps_1, false, items_per_thread, mult_per_warp>
        (twiddle_cos_a, twiddle_sin_a, d_twiddle_a, output_val, grad_val, nsteps_2, input_idx_1, 0);
      block_exchange<items_per_thread, mult_per_warp, smem_per_thread>(temp_storage, grad_val, threadIdx.x, thread_idx_2, nthreads);
      __syncthreads();
      block_exchange<items_per_thread, mult_per_warp, smem_per_thread>(temp_storage, output_val, threadIdx.x, thread_idx_2, nthreads);
      b_ortho_untied_backward<nsteps_2, false, items_per_thread, mult_per_warp>
        (twiddle_cos_a, twiddle_sin_a, d_twiddle_a, output_val, grad_val, 0, input_idx_2, nsteps_1);
      block_exchange<items_per_thread, mult_per_warp, smem_per_thread>(temp_storage, grad_val, thread_idx_2, threadIdx.x, nthreads);
    }
  }
  d_input_writer.save<items_per_thread, mult_per_warp>(grad_val, input_idx_1);
}

void butterfly_ortho_multiply_untied_backward_fast_cuda(const at::Tensor &twiddle_cos,
                                                        const at::Tensor &twiddle_sin,
                                                        const at::Tensor &output,
                                                        const at::Tensor &grad,
                                                        at::Tensor& d_twiddle,
                                                        at::Tensor& d_input,
                                                        bool increasing_stride) {
  int batch_size = output.size(0);
  const int nstack = output.size(1);
  const int n = output.size(2);
  const int log_n = int(log2((double) n));
  AT_DISPATCH_FLOATING_TYPES(output.scalar_type(), "butterfly_ortho_multiply_untied_backward_fast_cuda", [&] {
    const auto twiddle_cos_a = twiddle_cos.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>();
    const auto twiddle_sin_a = twiddle_sin.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>();
    const InputReader<scalar_t> output_reader(output);
    const InputReader<scalar_t> grad_reader(grad);
    auto d_twiddle_a = d_twiddle.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>();
    OutputWriter<scalar_t> d_input_writer(d_input);
    dim3 block(min(n, MAX_BLOCK_SIZE));
    dim3 grid(div_up(batch_size, ITEMS_PER_THREAD_ORTHO_BACKWARD[log_n - 1]), 1, nstack);
    auto stream = at::cuda::getCurrentCUDAStream();
    switch (log_n) {
      #define CASE_LOG_N(log_n_val) case log_n_val:                                                                                     \
      increasing_stride ? butterfly_ortho_multiply_untied_backward_fast_cuda_kernel<log_n_val, true>                                    \
        <<<grid, block, 0, stream>>>(twiddle_cos_a, twiddle_sin_a, output_reader, grad_reader, d_twiddle_a, d_input_writer, batch_size) \
        : butterfly_ortho_multiply_untied_backward_fast_cuda_kernel<log_n_val, false>                                                   \
        <<<grid, block, 0, stream>>>(twiddle_cos_a, twiddle_sin_a, output_reader, grad_reader, d_twiddle_a, d_input_writer, batch_size); break;
      // MAP(CASE_LOG_N, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)
      MAP(CASE_LOG_N, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
    }
  });
  // Have to keep this #undef outside the AT_DISPATCH_FLOATING_TYPES macro for it to work
  #undef CASE_LOG_N
  TORCH_CHECK(cudaGetLastError() == cudaSuccess,
     "butterfly_ortho_multiply_untied_backward_fast_cuda failed with error code ",
     cudaGetLastError());
}

template <int items_per_thread, int mult_per_warp=1, typename scalar_t>
__device__ __forceinline__ void diag_forward(const CudaAcsr<scalar_t, 3> diagonal_a,
                                             scalar_t input_val[mult_per_warp][items_per_thread],
                                             int diagonal_idx,
                                             int input_idx) {
  const int s = blockIdx.y + gridDim.y * blockIdx.z;  // For conv2d butterfly_ortho as well
  #pragma unroll
  for (int mult = 0; mult < mult_per_warp; mult++) {
    const scalar_t diag_val = diagonal_a[s][diagonal_idx][mult * warpSize + input_idx];
    #pragma unroll
    for (int item = 0; item < items_per_thread; item++) {
      input_val[mult][item] *= diag_val;
    }
  }
}

template <int log_n, int items_per_thread=ITEMS_PER_THREAD_ORTHO_FORWARD[log_n - 1],
            int min_blocks_per_mp=MIN_BLOCKS_PER_MP_ORTHO_FORWARD[log_n - 1],
            int max_smem_per_thread=items_per_thread, typename scalar_t>
// C10_LAUNCH_BOUNDS_2 supposedly takes min(1 << log_n, 1024)
// https://github.com/pytorch/pytorch/blob/v1.1.0/c10/macros/Macros.h
// However, it doesn't seem to work correctly so I have to take min explicitly.
C10_LAUNCH_BOUNDS_2(MIN_MACRO(1 << log_n, MAX_BLOCK_SIZE), min_blocks_per_mp)
__global__ void butterfly_odo_multiply_untied_forward_fast_cuda_kernel(const CudaAcsr<scalar_t, 3> twiddle_cos_a,
                                                                       const CudaAcsr<scalar_t, 3> twiddle_sin_a,
                                                                       const CudaAcsr<scalar_t, 3> diagonal_a,
                                                                       InputReader<scalar_t> input_reader,
                                                                       OutputWriter<scalar_t> output_writer,
                                                                       int batch_size,
                                                                       int nblocks) {
  constexpr int n = 1 << log_n;
  constexpr int nthreads = min_const(n, MAX_BLOCK_SIZE);
  constexpr int smem_limit = min_const(SMEM_PER_MP / min_blocks_per_mp, MAX_SMEM_PER_BLOCK);
  constexpr int smem_per_thread = min_const(max_smem_per_thread, items_per_thread,
                                            smem_limit / (nthreads * sizeof(scalar_t)));
  constexpr int mult_per_warp = n / nthreads;
  scalar_t input_val[mult_per_warp][items_per_thread];
  // const int input_idx_1 = (threadIdx.x % warpSize) + mult_per_warp * warpSize * (threadIdx.x / warpSize);
  const int input_idx_1 = (threadIdx.x & ((1 << 5) - 1)) + mult_per_warp * warpSize * (threadIdx.x >> 5);
  input_reader.load<items_per_thread, mult_per_warp>(input_val, input_idx_1);
  if (log_n <= 5) {
    for (int block = 0; block < nblocks; block++) {
      b_ortho_untied_forward<min_const(log_n, 5), false, items_per_thread>
        (twiddle_cos_a, twiddle_sin_a, input_val, block * 2 * log_n, input_idx_1, 0);
      diag_forward<items_per_thread>(diagonal_a, input_val, block, input_idx_1);
      b_ortho_untied_forward<min_const(log_n, 5), true, items_per_thread>
        (twiddle_cos_a, twiddle_sin_a, input_val, (block * 2 + 1) * log_n, input_idx_1, 0);
    }
  } else {
    __shared__ scalar_t temp_storage[nthreads * smem_per_thread];
    constexpr int nsteps_1 = log_n <= 10 ? 5 : log_n - 5;
    constexpr int nsteps_2 = max_const(log_n - nsteps_1, 1);  // Take max to avoid compiler's warning
    constexpr int log_nwarps = min_const(max_const(log_n - 5, 1), 5);  // Take max to avoid compiler's warning
    const int input_idx_2 = ((threadIdx.x & ((1 << log_nwarps) - 1)) << nsteps_1) + (threadIdx.x >> log_nwarps);
    const int thread_idx_2 = (threadIdx.x & ((1 << log_nwarps) - 1)) * warpSize + (threadIdx.x >> log_nwarps);
    block_exchange<items_per_thread, mult_per_warp, smem_per_thread>(temp_storage, input_val, threadIdx.x, thread_idx_2, nthreads);
    for (int block = 0; block < nblocks; block++) {
      b_ortho_untied_forward<nsteps_2, false, items_per_thread, mult_per_warp>
        (twiddle_cos_a, twiddle_sin_a, input_val, block * 2 * log_n, input_idx_2, nsteps_1);
      block_exchange<items_per_thread, mult_per_warp, smem_per_thread>(temp_storage, input_val, thread_idx_2, threadIdx.x, nthreads);
      b_ortho_untied_forward<nsteps_1, false, items_per_thread, mult_per_warp>
        (twiddle_cos_a, twiddle_sin_a, input_val, block * 2 * log_n + nsteps_2, input_idx_1, 0);
      diag_forward<items_per_thread, mult_per_warp>(diagonal_a, input_val, block, input_idx_1);
      b_ortho_untied_forward<nsteps_1, true, items_per_thread, mult_per_warp>
        (twiddle_cos_a, twiddle_sin_a, input_val, (block * 2 + 1) * log_n, input_idx_1, 0);
      block_exchange<items_per_thread, mult_per_warp, smem_per_thread>(temp_storage, input_val, threadIdx.x, thread_idx_2, nthreads);
      b_ortho_untied_forward<nsteps_2, true, items_per_thread, mult_per_warp>
        (twiddle_cos_a, twiddle_sin_a, input_val, (block * 2 + 1) * log_n + nsteps_1, input_idx_2, nsteps_1);
    }
    block_exchange<items_per_thread, mult_per_warp, smem_per_thread>(temp_storage, input_val, thread_idx_2, threadIdx.x, nthreads);
  }
  output_writer.save<items_per_thread, mult_per_warp>(input_val, input_idx_1);
}

void butterfly_odo_multiply_untied_forward_fast_cuda(const at::Tensor &twiddle_cos,
                                                     const at::Tensor &twiddle_sin,
                                                     const at::Tensor &diagonal,
                                                     const at::Tensor &input,
                                                     at::Tensor &output) {
  int batch_size = input.size(0);
  const int nstack = input.size(1);
  const int n = input.size(2);
  const int log_n = int(log2((double) n));
  const int nblocks = twiddle_cos.size(1) / (2 * log_n);
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "butterfly_odo_multiply_untied_forward_fast_cuda", [&] {
    const auto twiddle_cos_a = twiddle_cos.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>();
    const auto twiddle_sin_a = twiddle_sin.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>();
    const auto diagonal_a = diagonal.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>();
    const InputReader<scalar_t> input_reader(input);
    OutputWriter<scalar_t> output_writer(output);
    dim3 block(min(n, MAX_BLOCK_SIZE));
    dim3 grid(div_up(batch_size, ITEMS_PER_THREAD_ORTHO_FORWARD[log_n - 1]), 1, nstack);
    auto stream = at::cuda::getCurrentCUDAStream();
    switch (log_n) {
      #define CASE_LOG_N(log_n_val) case log_n_val:                     \
      butterfly_odo_multiply_untied_forward_fast_cuda_kernel<log_n_val> \
        <<<grid, block, 0, stream>>>(twiddle_cos_a, twiddle_sin_a, diagonal_a, input_reader, output_writer, batch_size, nblocks); break;
      MAP(CASE_LOG_N, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)
    }
  });
  // Have to keep this #undef outside the AT_DISPATCH_FLOATING_TYPES macro for it to work
  #undef CASE_LOG_N
  TORCH_CHECK(cudaGetLastError() == cudaSuccess,
     "butterfly_odo_multiply_untied_forward_fast_cuda failed with error code ",
     cudaGetLastError());
}

template <int items_per_thread, int mult_per_warp=1, typename scalar_t,
            typename accscalar_t=at::acc_type<scalar_t, true>>
__device__ __forceinline__ void diag_backward(const CudaAcsr<scalar_t, 3> diagonal_a,
                                              CudaAcsr<scalar_t, 3> d_diagonal_a,
                                              scalar_t output_val[mult_per_warp][items_per_thread],
                                              scalar_t grad_val[mult_per_warp][items_per_thread],
                                              int diagonal_idx,
                                              int input_idx) {
  const int s = blockIdx.y + gridDim.y * blockIdx.z;  // For conv2d butterfly_ortho as well
  #pragma unroll
  for (int mult = 0; mult < mult_per_warp; mult++) {
    const scalar_t diag_val = diagonal_a[s][diagonal_idx][mult * warpSize + input_idx];
    accscalar_t d_diag_val = 0;
    #pragma unroll
    for (int item = 0; item < items_per_thread; item++) {
      output_val[mult][item] /= diag_val;
      d_diag_val += output_val[mult][item] * grad_val[mult][item];
      grad_val[mult][item] *= diag_val;
    }
    atomicAdd(&d_diagonal_a[s][diagonal_idx][mult * warpSize + input_idx], d_diag_val);
  }
}

template <int log_n, int items_per_thread=ITEMS_PER_THREAD_ORTHO_BACKWARD[log_n - 1],
            int min_blocks_per_mp=MIN_BLOCKS_PER_MP_ORTHO_BACKWARD[log_n - 1],
            int max_smem_per_thread=items_per_thread, typename scalar_t>
C10_LAUNCH_BOUNDS_2(MIN_MACRO(1 << log_n, MAX_BLOCK_SIZE), min_blocks_per_mp)
__global__ void butterfly_odo_multiply_untied_backward_fast_cuda_kernel(const CudaAcsr<scalar_t, 3> twiddle_cos_a,
                                                                        const CudaAcsr<scalar_t, 3> twiddle_sin_a,
                                                                        const CudaAcsr<scalar_t, 3> diagonal_a,
                                                                        InputReader<scalar_t> output_reader,
                                                                        InputReader<scalar_t> grad_reader,
                                                                        CudaAcsr<scalar_t, 3> d_twiddle_a,
                                                                        CudaAcsr<scalar_t, 3> d_diagonal_a,
                                                                        OutputWriter<scalar_t> d_input_writer,
                                                                        int batch_size,
                                                                        int nblocks) {
  constexpr int n = 1 << log_n;
  constexpr int nthreads = min_const(n, MAX_BLOCK_SIZE);
  constexpr int smem_limit = min_const(SMEM_PER_MP / min_blocks_per_mp, MAX_SMEM_PER_BLOCK);
  constexpr int smem_per_thread = min_const(max_smem_per_thread, items_per_thread,
                                            smem_limit / (nthreads * sizeof(scalar_t)));
  constexpr int mult_per_warp = n / nthreads;
  scalar_t output_val[mult_per_warp][items_per_thread];
  scalar_t grad_val[mult_per_warp][items_per_thread];
  // const int input_idx_1 = (threadIdx.x % warpSize) + mult_per_warp * warpSize * (threadIdx.x / warpSize);
  const int input_idx_1 = (threadIdx.x & ((1 << 5) - 1)) + mult_per_warp * warpSize * (threadIdx.x >> 5);
  output_reader.load<items_per_thread, mult_per_warp>(output_val, input_idx_1);
  if (log_n <= 5) {
    grad_reader.load<items_per_thread, mult_per_warp>(grad_val, input_idx_1);
    for (int block = nblocks - 1; block >= 0; block--) {
      b_ortho_untied_backward<min_const(log_n, 5), true, items_per_thread, mult_per_warp>
        (twiddle_cos_a, twiddle_sin_a, d_twiddle_a, output_val, grad_val, (block * 2 + 1) * log_n, input_idx_1, 0);
      diag_backward<items_per_thread>(diagonal_a, d_diagonal_a, output_val, grad_val, block, input_idx_1);
      b_ortho_untied_backward<min_const(log_n, 5), false, items_per_thread, mult_per_warp>
        (twiddle_cos_a, twiddle_sin_a, d_twiddle_a, output_val, grad_val, block * 2 * log_n, input_idx_1, 0);
    }
  } else {
    __shared__ scalar_t temp_storage[nthreads * smem_per_thread];
    // constexpr int nsteps_1 = div_up_const(log_n, 2);
    constexpr int nsteps_1 = log_n <= 10 ? 5 : log_n - 5;
    constexpr int nsteps_2 = max_const(log_n - nsteps_1, 1);  // Take max to avoid compiler's warning
    constexpr int log_nwarps = min_const(max_const(log_n - 5, 1), 5);  // Take max to avoid compiler's warning
    const int input_idx_2 = ((threadIdx.x & ((1 << log_nwarps) - 1)) << nsteps_1) + (threadIdx.x >> log_nwarps);
    const int thread_idx_2 = (threadIdx.x & ((1 << log_nwarps) - 1)) * warpSize + (threadIdx.x >> log_nwarps);
    block_exchange<items_per_thread, mult_per_warp, smem_per_thread>(temp_storage, output_val, threadIdx.x, thread_idx_2, nthreads);
    grad_reader.load<items_per_thread, mult_per_warp>(grad_val, input_idx_2);
    for (int block = nblocks - 1; block >= 0; block--) {
      b_ortho_untied_backward<nsteps_2, true, items_per_thread, mult_per_warp>
        (twiddle_cos_a, twiddle_sin_a, d_twiddle_a, output_val, grad_val, (block * 2 + 1) * log_n + nsteps_1, input_idx_2, nsteps_1);
      block_exchange<items_per_thread, mult_per_warp, smem_per_thread>(temp_storage, grad_val, thread_idx_2, threadIdx.x, nthreads);
      __syncthreads();
      block_exchange<items_per_thread, mult_per_warp, smem_per_thread>(temp_storage, output_val, thread_idx_2, threadIdx.x, nthreads);
      b_ortho_untied_backward<nsteps_1, true, items_per_thread, mult_per_warp>
        (twiddle_cos_a, twiddle_sin_a, d_twiddle_a, output_val, grad_val, (block * 2 + 1) * log_n, input_idx_1, 0);
      diag_backward<items_per_thread, mult_per_warp>(diagonal_a, d_diagonal_a, output_val, grad_val, block, input_idx_1);
      b_ortho_untied_backward<nsteps_1, false, items_per_thread, mult_per_warp>
        (twiddle_cos_a, twiddle_sin_a, d_twiddle_a, output_val, grad_val, block * 2 * log_n + nsteps_2, input_idx_1, 0);
      block_exchange<items_per_thread, mult_per_warp, smem_per_thread>(temp_storage, grad_val, threadIdx.x, thread_idx_2, nthreads);
      __syncthreads();
      block_exchange<items_per_thread, mult_per_warp, smem_per_thread>(temp_storage, output_val, threadIdx.x, thread_idx_2, nthreads);
      b_ortho_untied_backward<nsteps_2, false, items_per_thread, mult_per_warp>
        (twiddle_cos_a, twiddle_sin_a, d_twiddle_a, output_val, grad_val, block * 2 * log_n, input_idx_2, nsteps_1);
    }
    block_exchange<items_per_thread, mult_per_warp, smem_per_thread>(temp_storage, grad_val, thread_idx_2, threadIdx.x, nthreads);
  }
  d_input_writer.save<items_per_thread, mult_per_warp>(grad_val, input_idx_1);
}

void butterfly_odo_multiply_untied_backward_fast_cuda(const at::Tensor &twiddle_cos,
                                                      const at::Tensor &twiddle_sin,
                                                      const at::Tensor &diagonal,
                                                      const at::Tensor &output,
                                                      const at::Tensor &grad,
                                                      at::Tensor& d_twiddle,
                                                      at::Tensor& d_diagonal,
                                                      at::Tensor& d_input) {
  int batch_size = output.size(0);
  const int nstack = output.size(1);
  const int n = output.size(2);
  const int log_n = int(log2((double) n));
  const int nblocks = twiddle_cos.size(1) / (2 * log_n);
  AT_DISPATCH_FLOATING_TYPES(output.scalar_type(), "butterfly_odo_multiply_untied_backward_fast_cuda", [&] {
    const auto twiddle_cos_a = twiddle_cos.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>();
    const auto twiddle_sin_a = twiddle_sin.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>();
    const auto diagonal_a = diagonal.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>();
    const InputReader<scalar_t> output_reader(output);
    const InputReader<scalar_t> grad_reader(grad);
    auto d_twiddle_a = d_twiddle.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>();
    auto d_diagonal_a = d_diagonal.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>();
    OutputWriter<scalar_t> d_input_writer(d_input);
    dim3 block(min(n, MAX_BLOCK_SIZE));
    dim3 grid(div_up(batch_size, ITEMS_PER_THREAD_ORTHO_BACKWARD[log_n - 1]), 1, nstack);
    auto stream = at::cuda::getCurrentCUDAStream();
    switch (log_n) {
      #define CASE_LOG_N(log_n_val) case log_n_val:                      \
      butterfly_odo_multiply_untied_backward_fast_cuda_kernel<log_n_val> \
        <<<grid, block, 0, stream>>>(twiddle_cos_a, twiddle_sin_a, diagonal_a, output_reader, grad_reader, d_twiddle_a, d_diagonal_a, d_input_writer, batch_size, nblocks); break;
      // MAP(CASE_LOG_N, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)
      MAP(CASE_LOG_N, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
    }
  });
  // Have to keep this #undef outside the AT_DISPATCH_FLOATING_TYPES macro for it to work
  #undef CASE_LOG_N
  TORCH_CHECK(cudaGetLastError() == cudaSuccess,
     "butterfly_odo_multiply_untied_backward_fast_cuda failed with error code ",
     cudaGetLastError());
}

template <int items_per_thread, int mult_per_warp=1, typename scalar_t,
            typename accscalar_t=at::acc_type<scalar_t, true>>
__device__ __forceinline__ void diag_backward_with_input(const CudaAcsr<scalar_t, 3> diagonal_a,
                                                         CudaAcsr<scalar_t, 3> d_diagonal_a,
                                                         scalar_t input_val[mult_per_warp][items_per_thread],
                                                         scalar_t grad_val[mult_per_warp][items_per_thread],
                                                         int diagonal_idx,
                                                         int input_idx) {
  const int s = blockIdx.y + gridDim.y * blockIdx.z;  // For conv2d butterfly_ortho as well
  #pragma unroll
  for (int mult = 0; mult < mult_per_warp; mult++) {
    const scalar_t diag_val = diagonal_a[s][diagonal_idx][mult * warpSize + input_idx];
    accscalar_t d_diag_val = 0;
    #pragma unroll
    for (int item = 0; item < items_per_thread; item++) {
      d_diag_val += input_val[mult][item] * grad_val[mult][item];
      grad_val[mult][item] *= diag_val;
    }
    atomicAdd(&d_diagonal_a[s][diagonal_idx][mult * warpSize + input_idx], d_diag_val);
  }
}

template <int log_n, int items_per_thread=ITEMS_PER_THREAD_ORTHO_BACKWARD[log_n - 1],
            int min_blocks_per_mp=MIN_BLOCKS_PER_MP_ORTHO_BACKWARD[log_n - 1],
            int max_smem_per_thread=items_per_thread, typename scalar_t>
C10_LAUNCH_BOUNDS_2(MIN_MACRO(1 << log_n, MAX_BLOCK_SIZE), min_blocks_per_mp)
__global__ void butterfly_odo_multiply_untied_forward_backward_fast_cuda_kernel(const CudaAcsr<scalar_t, 3> twiddle_cos_a,
                                                                                const CudaAcsr<scalar_t, 3> twiddle_sin_a,
                                                                                const CudaAcsr<scalar_t, 3> diagonal_a,
                                                                                InputReader<scalar_t> input_reader,
                                                                                InputReader<scalar_t> grad_reader,
                                                                                IntermediateStorage<scalar_t> inter_storage,
                                                                                CudaAcsr<scalar_t, 3> d_twiddle_a,
                                                                                CudaAcsr<scalar_t, 3> d_diagonal_a,
                                                                                OutputWriter<scalar_t> d_input_writer,
                                                                                int batch_size,
                                                                                int nblocks) {
  constexpr int n = 1 << log_n;
  constexpr int nthreads = min_const(n, MAX_BLOCK_SIZE);
  constexpr int smem_limit = min_const(SMEM_PER_MP / min_blocks_per_mp, MAX_SMEM_PER_BLOCK);
  constexpr int smem_per_thread = min_const(max_smem_per_thread, items_per_thread,
                                            smem_limit / (nthreads * sizeof(scalar_t)));
  constexpr int mult_per_warp = n / nthreads;
  scalar_t input_val[mult_per_warp][items_per_thread];
  scalar_t grad_val[mult_per_warp][items_per_thread];
  // const int input_idx_1 = (threadIdx.x % warpSize) + mult_per_warp * warpSize * (threadIdx.x / warpSize);
  const int input_idx_1 = (threadIdx.x & ((1 << 5) - 1)) + mult_per_warp * warpSize * (threadIdx.x >> 5);
  input_reader.load<items_per_thread, mult_per_warp>(input_val, input_idx_1);
  if (log_n <= 5) {
    for (int block = 0; block < nblocks; block++) {
      b_ortho_untied_forward<min_const(log_n, 5), false, items_per_thread>
        (twiddle_cos_a, twiddle_sin_a, input_val, block * 2 * log_n, input_idx_1, 0);
      inter_storage.save<items_per_thread>(input_val, input_idx_1, block);
      diag_forward<items_per_thread>(diagonal_a, input_val, block, input_idx_1);
      b_ortho_untied_forward<min_const(log_n, 5), true, items_per_thread>
        (twiddle_cos_a, twiddle_sin_a, input_val, (block * 2 + 1) * log_n, input_idx_1, 0);
    }
    grad_reader.load<items_per_thread, mult_per_warp>(grad_val, input_idx_1);
    for (int block = nblocks - 1; block >= 0; block--) {
      b_ortho_untied_backward<min_const(log_n, 5), true, items_per_thread, mult_per_warp>
        (twiddle_cos_a, twiddle_sin_a, d_twiddle_a, input_val, grad_val, (block * 2 + 1) * log_n, input_idx_1, 0);
      inter_storage.load<items_per_thread>(input_val, input_idx_1, block);
      diag_backward_with_input<items_per_thread>(diagonal_a, d_diagonal_a, input_val, grad_val, block, input_idx_1);
      b_ortho_untied_backward<min_const(log_n, 5), false, items_per_thread, mult_per_warp>
        (twiddle_cos_a, twiddle_sin_a, d_twiddle_a, input_val, grad_val, block * 2 * log_n, input_idx_1, 0);
    }
  } else {
    __shared__ scalar_t temp_storage[nthreads * smem_per_thread];
    // constexpr int nsteps_1 = div_up_const(log_n, 2);
    constexpr int nsteps_1 = log_n <= 10 ? 5 : log_n - 5;
    constexpr int nsteps_2 = max_const(log_n - nsteps_1, 1);  // Take max to avoid compiler's warning
    constexpr int log_nwarps = min_const(max_const(log_n - 5, 1), 5);  // Take max to avoid compiler's warning
    const int input_idx_2 = ((threadIdx.x & ((1 << log_nwarps) - 1)) << nsteps_1) + (threadIdx.x >> log_nwarps);
    const int thread_idx_2 = (threadIdx.x & ((1 << log_nwarps) - 1)) * warpSize + (threadIdx.x >> log_nwarps);
    block_exchange<items_per_thread, mult_per_warp, smem_per_thread>(temp_storage, input_val, threadIdx.x, thread_idx_2, nthreads);
    for (int block = 0; block < nblocks; block++) {
      b_ortho_untied_forward<nsteps_2, false, items_per_thread, mult_per_warp>
        (twiddle_cos_a, twiddle_sin_a, input_val, block * 2 * log_n, input_idx_2, nsteps_1);
      block_exchange<items_per_thread, mult_per_warp, smem_per_thread>(temp_storage, input_val, thread_idx_2, threadIdx.x, nthreads);
      b_ortho_untied_forward<nsteps_1, false, items_per_thread, mult_per_warp>
        (twiddle_cos_a, twiddle_sin_a, input_val, block * 2 * log_n + nsteps_2, input_idx_1, 0);
      inter_storage.save<items_per_thread, mult_per_warp>(input_val, input_idx_1, block);
      diag_forward<items_per_thread, mult_per_warp>(diagonal_a, input_val, block, input_idx_1);
      b_ortho_untied_forward<nsteps_1, true, items_per_thread, mult_per_warp>
        (twiddle_cos_a, twiddle_sin_a, input_val, (block * 2 + 1) * log_n, input_idx_1, 0);
      block_exchange<items_per_thread, mult_per_warp, smem_per_thread>(temp_storage, input_val, threadIdx.x, thread_idx_2, nthreads);
      b_ortho_untied_forward<nsteps_2, true, items_per_thread, mult_per_warp>
        (twiddle_cos_a, twiddle_sin_a, input_val, (block * 2 + 1) * log_n + nsteps_1, input_idx_2, nsteps_1);
    }
    grad_reader.load<items_per_thread, mult_per_warp>(grad_val, input_idx_2);
    for (int block = nblocks - 1; block >= 0; block--) {
      b_ortho_untied_backward<nsteps_2, true, items_per_thread, mult_per_warp>
        (twiddle_cos_a, twiddle_sin_a, d_twiddle_a, input_val, grad_val, (block * 2 + 1) * log_n + nsteps_1, input_idx_2, nsteps_1);
      block_exchange<items_per_thread, mult_per_warp, smem_per_thread>(temp_storage, grad_val, thread_idx_2, threadIdx.x, nthreads);
      __syncthreads();
      block_exchange<items_per_thread, mult_per_warp, smem_per_thread>(temp_storage, input_val, thread_idx_2, threadIdx.x, nthreads);
      b_ortho_untied_backward<nsteps_1, true, items_per_thread, mult_per_warp>
        (twiddle_cos_a, twiddle_sin_a, d_twiddle_a, input_val, grad_val, (block * 2 + 1) * log_n, input_idx_1, 0);
      inter_storage.load<items_per_thread, mult_per_warp>(input_val, input_idx_1, block);
      diag_backward_with_input<items_per_thread, mult_per_warp>(diagonal_a, d_diagonal_a, input_val, grad_val, block, input_idx_1);
      b_ortho_untied_backward<nsteps_1, false, items_per_thread, mult_per_warp>
        (twiddle_cos_a, twiddle_sin_a, d_twiddle_a, input_val, grad_val, block * 2 * log_n + nsteps_2, input_idx_1, 0);
      block_exchange<items_per_thread, mult_per_warp, smem_per_thread>(temp_storage, grad_val, threadIdx.x, thread_idx_2, nthreads);
      __syncthreads();
      block_exchange<items_per_thread, mult_per_warp, smem_per_thread>(temp_storage, input_val, threadIdx.x, thread_idx_2, nthreads);
      b_ortho_untied_backward<nsteps_2, false, items_per_thread, mult_per_warp>
        (twiddle_cos_a, twiddle_sin_a, d_twiddle_a, input_val, grad_val, block * 2 * log_n, input_idx_2, nsteps_1);
    }
    block_exchange<items_per_thread, mult_per_warp, smem_per_thread>(temp_storage, grad_val, thread_idx_2, threadIdx.x, nthreads);
  }
  d_input_writer.save<items_per_thread, mult_per_warp>(grad_val, input_idx_1);
}

void butterfly_odo_multiply_untied_forward_backward_fast_cuda(const at::Tensor &twiddle_cos,
                                                              const at::Tensor &twiddle_sin,
                                                              const at::Tensor &diagonal,
                                                              const at::Tensor &input,
                                                              const at::Tensor &grad,
                                                              at::Tensor& d_twiddle,
                                                              at::Tensor& d_diagonal,
                                                              at::Tensor& d_input) {
  int batch_size = input.size(0);
  const int nstack = input.size(1);
  const int n = input.size(2);
  const int log_n = int(log2((double) n));
  const int nblocks = twiddle_cos.size(1) / (2 * log_n);
  auto intermediate_storage = at::empty({nblocks, batch_size, nstack, n},
                                        at::dtype(input.dtype()).device(input.device()));
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "butterfly_odo_multiply_untied_forward_backward_fast_cuda", [&] {
    const auto twiddle_cos_a = twiddle_cos.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>();
    const auto twiddle_sin_a = twiddle_sin.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>();
    const auto diagonal_a = diagonal.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>();
    const InputReader<scalar_t> input_reader(input);
    const InputReader<scalar_t> grad_reader(grad);
    IntermediateStorage<scalar_t> inter_storage(intermediate_storage);
    auto d_twiddle_a = d_twiddle.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>();
    auto d_diagonal_a = d_diagonal.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>();
    OutputWriter<scalar_t> d_input_writer(d_input);
    dim3 block(min(n, MAX_BLOCK_SIZE));
    dim3 grid(div_up(batch_size, ITEMS_PER_THREAD_ORTHO_BACKWARD[log_n - 1]), 1, nstack);
    auto stream = at::cuda::getCurrentCUDAStream();
    switch (log_n) {
      #define CASE_LOG_N(log_n_val) case log_n_val:                      \
      butterfly_odo_multiply_untied_forward_backward_fast_cuda_kernel<log_n_val> \
        <<<grid, block, 0, stream>>>(twiddle_cos_a, twiddle_sin_a, diagonal_a, input_reader, grad_reader, inter_storage, d_twiddle_a, d_diagonal_a, d_input_writer, batch_size, nblocks); break;
      // MAP(CASE_LOG_N, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)
      MAP(CASE_LOG_N, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
    }
  });
  // Have to keep this #undef outside the AT_DISPATCH_FLOATING_TYPES macro for it to work
  #undef CASE_LOG_N
  TORCH_CHECK(cudaGetLastError() == cudaSuccess,
     "butterfly_odo_multiply_untied_forward_backward_fast_cuda failed with error code ",
     cudaGetLastError());
}

#if BFLY_BENCHMARK
void butterfly_odo_multiply_untied_forward_fast_cuda_benchmark(const at::Tensor &twiddle_cos,
                                                               const at::Tensor &twiddle_sin,
                                                               const at::Tensor &diagonal,
                                                               const at::Tensor &input,
                                                               at::Tensor &output) {
  int batch_size = input.size(0);
  const int nstack = input.size(1);
  const int n = input.size(2);
  const int log_n = int(log2((double) n));
  const int nblocks = twiddle_cos.size(1) / (2 * log_n);
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "butterfly_odo_multiply_untied_forward_fast_cuda_benchmark", [&] {
    const auto twiddle_cos_a = twiddle_cos.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>();
    const auto twiddle_sin_a = twiddle_sin.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>();
    const auto diagonal_a = diagonal.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>();
    const InputReader<scalar_t> input_reader(input);
    OutputWriter<scalar_t> output_writer(output);
    dim3 block(min(n, MAX_BLOCK_SIZE));
    auto stream = at::cuda::getCurrentCUDAStream();
    switch (log_n) {
      case 9:
        #define CASE_IPT_9(items_per_thread_val) do {                                      \
        dim3 grid(div_up(batch_size, items_per_thread_val), 1, nstack);                    \
        butterfly_odo_multiply_untied_forward_fast_cuda_kernel<9, items_per_thread_val, 1> \
          <<<grid, block, 0, stream>>>(twiddle_cos_a, twiddle_sin_a, diagonal_a, input_reader, output_writer, batch_size, nblocks); \
        butterfly_odo_multiply_untied_forward_fast_cuda_kernel<9, items_per_thread_val, 2> \
          <<<grid, block, 0, stream>>>(twiddle_cos_a, twiddle_sin_a, diagonal_a, input_reader, output_writer, batch_size, nblocks); \
        } while (0);
        // MAP(CASE_IPT_9, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 16);
        // MAP(CASE_IPT_9, 1, 2, 4, 6, 8, 12, 16);
        MAP(CASE_IPT_9, 6, 8, 12, 16);
        break;
      case 10:
        #define CASE_IPT_10(items_per_thread_val) do {                                      \
        dim3 grid(div_up(batch_size, items_per_thread_val), 1, nstack);                     \
        butterfly_odo_multiply_untied_forward_fast_cuda_kernel<10, items_per_thread_val, 1> \
          <<<grid, block, 0, stream>>>(twiddle_cos_a, twiddle_sin_a, diagonal_a, input_reader, output_writer, batch_size, nblocks); \
        butterfly_odo_multiply_untied_forward_fast_cuda_kernel<10, items_per_thread_val, 2> \
          <<<grid, block, 0, stream>>>(twiddle_cos_a, twiddle_sin_a, diagonal_a, input_reader, output_writer, batch_size, nblocks); \
        } while (0);
        // MAP(CASE_IPT_10, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 16);
        // MAP(CASE_IPT_10, 1, 2, 4, 6, 8, 12, 16);
        MAP(CASE_IPT_10, 6, 8, 12, 16);
        break;
      case 11:
        #define CASE_IPT_11(items_per_thread_val) do {                                      \
        dim3 grid(div_up(batch_size, items_per_thread_val), 1, nstack);                     \
        butterfly_odo_multiply_untied_forward_fast_cuda_kernel<11, items_per_thread_val, 1> \
          <<<grid, block, 0, stream>>>(twiddle_cos_a, twiddle_sin_a, diagonal_a, input_reader, output_writer, batch_size, nblocks); \
        butterfly_odo_multiply_untied_forward_fast_cuda_kernel<11, items_per_thread_val, 2> \
          <<<grid, block, 0, stream>>>(twiddle_cos_a, twiddle_sin_a, diagonal_a, input_reader, output_writer, batch_size, nblocks); \
        } while (0);
        // MAP(CASE_IPT_11, 1, 2, 4, 6, 8, 12, 16);
        MAP(CASE_IPT_11, 6, 8, 12, 16);
        break;
    }
  });
  // Have to keep this #undef outside the AT_DISPATCH_FLOATING_TYPES macro for it to work
  #undef CASE_IPT_9
  #undef CASE_IPT_10
  #undef CASE_IPT_11
  TORCH_CHECK(cudaGetLastError() == cudaSuccess,
     "butterfly_odo_multiply_untied_forward_fast_cuda_benchmark failed with error code ",
     cudaGetLastError());
}

void butterfly_odo_multiply_untied_backward_fast_cuda_benchmark(const at::Tensor &twiddle_cos,
                                                                const at::Tensor &twiddle_sin,
                                                                const at::Tensor &diagonal,
                                                                const at::Tensor &output,
                                                                const at::Tensor &grad,
                                                                at::Tensor& d_twiddle,
                                                                at::Tensor& d_diagonal,
                                                                at::Tensor& d_input) {
  int batch_size = output.size(0);
  const int nstack = output.size(1);
  const int n = output.size(2);
  const int log_n = int(log2((double) n));
  const int nblocks = twiddle_cos.size(1) / (2 * log_n);
  AT_DISPATCH_FLOATING_TYPES(output.scalar_type(), "butterfly_odo_multiply_untied_backward_fast_cuda_benchmark", [&] {
    const auto twiddle_cos_a = twiddle_cos.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>();
    const auto twiddle_sin_a = twiddle_sin.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>();
    const auto diagonal_a = diagonal.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>();
    const InputReader<scalar_t> output_reader(output);
    const InputReader<scalar_t> grad_reader(grad);
    auto d_twiddle_a = d_twiddle.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>();
    auto d_diagonal_a = d_diagonal.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>();
    OutputWriter<scalar_t> d_input_writer(d_input);
    dim3 block(min(n, MAX_BLOCK_SIZE));
    auto stream = at::cuda::getCurrentCUDAStream();
    switch (log_n) {
      case 9:
        #define CASE_IPT_9(items_per_thread_val) do {                                       \
        dim3 grid(div_up(batch_size, items_per_thread_val), 1, nstack);                     \
        butterfly_odo_multiply_untied_backward_fast_cuda_kernel<9, items_per_thread_val, 1> \
          <<<grid, block, 0, stream>>>(twiddle_cos_a, twiddle_sin_a, diagonal_a, output_reader, grad_reader, d_twiddle_a, d_diagonal_a, d_input_writer, batch_size, nblocks); \
        butterfly_odo_multiply_untied_backward_fast_cuda_kernel<9, items_per_thread_val, 2> \
          <<<grid, block, 0, stream>>>(twiddle_cos_a, twiddle_sin_a, diagonal_a, output_reader, grad_reader, d_twiddle_a, d_diagonal_a, d_input_writer, batch_size, nblocks); \
        butterfly_odo_multiply_untied_backward_fast_cuda_kernel<9, items_per_thread_val, 3> \
          <<<grid, block, 0, stream>>>(twiddle_cos_a, twiddle_sin_a, diagonal_a, output_reader, grad_reader, d_twiddle_a, d_diagonal_a, d_input_writer, batch_size, nblocks); \
        butterfly_odo_multiply_untied_backward_fast_cuda_kernel<9, items_per_thread_val, 4> \
          <<<grid, block, 0, stream>>>(twiddle_cos_a, twiddle_sin_a, diagonal_a, output_reader, grad_reader, d_twiddle_a, d_diagonal_a, d_input_writer, batch_size, nblocks); \
        } while(0);
        // MAP(CASE_IPT_9, 1, 2, 4, 6, 8, 12, 16, 24);
        break;
      case 10:
        #define CASE_IPT_10(items_per_thread_val) do {                                       \
        dim3 grid(div_up(batch_size, items_per_thread_val), 1, nstack);                      \
        butterfly_odo_multiply_untied_backward_fast_cuda_kernel<10, items_per_thread_val, 1> \
          <<<grid, block, 0, stream>>>(twiddle_cos_a, twiddle_sin_a, diagonal_a, output_reader, grad_reader, d_twiddle_a, d_diagonal_a, d_input_writer, batch_size, nblocks); \
        butterfly_odo_multiply_untied_backward_fast_cuda_kernel<10, items_per_thread_val, 2> \
          <<<grid, block, 0, stream>>>(twiddle_cos_a, twiddle_sin_a, diagonal_a, output_reader, grad_reader, d_twiddle_a, d_diagonal_a, d_input_writer, batch_size, nblocks); \
        butterfly_odo_multiply_untied_backward_fast_cuda_kernel<10, items_per_thread_val, 3> \
          <<<grid, block, 0, stream>>>(twiddle_cos_a, twiddle_sin_a, diagonal_a, output_reader, grad_reader, d_twiddle_a, d_diagonal_a, d_input_writer, batch_size, nblocks); \
        butterfly_odo_multiply_untied_backward_fast_cuda_kernel<10, items_per_thread_val, 4> \
          <<<grid, block, 0, stream>>>(twiddle_cos_a, twiddle_sin_a, diagonal_a, output_reader, grad_reader, d_twiddle_a, d_diagonal_a, d_input_writer, batch_size, nblocks); \
        } while(0);
        // MAP(CASE_IPT_10, 1, 2, 4, 6, 8, 12, 16, 24);
        break;
    }
  });
  // Have to keep this #undef outside the AT_DISPATCH_FLOATING_TYPES macro for it to work
  #undef CASE_IPT_9
  #undef CASE_IPT_10
  TORCH_CHECK(cudaGetLastError() == cudaSuccess,
     "butterfly_odo_multiply_untied_backward_fast_cuda_benchmark failed with error code ",
     cudaGetLastError());
}

void butterfly_odo_multiply_untied_forward_backward_fast_cuda_benchmark(const at::Tensor &twiddle_cos,
                                                                        const at::Tensor &twiddle_sin,
                                                                        const at::Tensor &diagonal,
                                                                        const at::Tensor &input,
                                                                        const at::Tensor &grad,
                                                                        at::Tensor& d_twiddle,
                                                                        at::Tensor& d_diagonal,
                                                                        at::Tensor& d_input) {
  int batch_size = input.size(0);
  const int nstack = input.size(1);
  const int n = input.size(2);
  const int log_n = int(log2((double) n));
  const int nblocks = twiddle_cos.size(1) / (2 * log_n);
  auto intermediate_storage = at::empty({nblocks, batch_size, nstack, n},
                                        at::dtype(input.dtype()).device(input.device()));
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "butterfly_odo_multiply_untied_forward_backward_fast_cuda_benchmark", [&] {
    const auto twiddle_cos_a = twiddle_cos.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>();
    const auto twiddle_sin_a = twiddle_sin.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>();
    const auto diagonal_a = diagonal.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>();
    const InputReader<scalar_t> input_reader(input);
    const InputReader<scalar_t> grad_reader(grad);
    IntermediateStorage<scalar_t> inter_storage(intermediate_storage);
    auto d_twiddle_a = d_twiddle.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>();
    auto d_diagonal_a = d_diagonal.packed_accessor<scalar_t, 3, at::RestrictPtrTraits, int32_t>();
    OutputWriter<scalar_t> d_input_writer(d_input);
    dim3 block(min(n, MAX_BLOCK_SIZE));
    auto stream = at::cuda::getCurrentCUDAStream();
    switch (log_n) {
      case 9:
        #define CASE_IPT_9(items_per_thread_val) do {                                               \
        dim3 grid(div_up(batch_size, items_per_thread_val), 1, nstack);                             \
        butterfly_odo_multiply_untied_forward_backward_fast_cuda_kernel<9, items_per_thread_val, 1> \
          <<<grid, block, 0, stream>>>(twiddle_cos_a, twiddle_sin_a, diagonal_a, input_reader, grad_reader, inter_storage, d_twiddle_a, d_diagonal_a, d_input_writer, batch_size, nblocks); \
        butterfly_odo_multiply_untied_forward_backward_fast_cuda_kernel<9, items_per_thread_val, 2> \
          <<<grid, block, 0, stream>>>(twiddle_cos_a, twiddle_sin_a, diagonal_a, input_reader, grad_reader, inter_storage, d_twiddle_a, d_diagonal_a, d_input_writer, batch_size, nblocks); \
        } while(0);
        // MAP(CASE_IPT_9, 1, 2, 4, 6, 8, 12, 16, 24);
        MAP(CASE_IPT_9, 6, 8, 12, 16);
        break;
      case 10:
        #define CASE_IPT_10(items_per_thread_val) do {                                               \
        dim3 grid(div_up(batch_size, items_per_thread_val), 1, nstack);                              \
        butterfly_odo_multiply_untied_forward_backward_fast_cuda_kernel<10, items_per_thread_val, 1> \
          <<<grid, block, 0, stream>>>(twiddle_cos_a, twiddle_sin_a, diagonal_a, input_reader, grad_reader, inter_storage, d_twiddle_a, d_diagonal_a, d_input_writer, batch_size, nblocks); \
        butterfly_odo_multiply_untied_forward_backward_fast_cuda_kernel<10, items_per_thread_val, 2> \
          <<<grid, block, 0, stream>>>(twiddle_cos_a, twiddle_sin_a, diagonal_a, input_reader, grad_reader, inter_storage, d_twiddle_a, d_diagonal_a, d_input_writer, batch_size, nblocks); \
        } while(0);
        // MAP(CASE_IPT_10, 1, 2, 4, 6, 8, 12, 16, 24);
        MAP(CASE_IPT_9, 6, 8, 12, 16);
        break;
      case 11:
        #define CASE_IPT_11(items_per_thread_val) do {                                               \
        dim3 grid(div_up(batch_size, items_per_thread_val), 1, nstack);                              \
        butterfly_odo_multiply_untied_forward_backward_fast_cuda_kernel<11, items_per_thread_val, 1> \
          <<<grid, block, 0, stream>>>(twiddle_cos_a, twiddle_sin_a, diagonal_a, input_reader, grad_reader, inter_storage, d_twiddle_a, d_diagonal_a, d_input_writer, batch_size, nblocks); \
        butterfly_odo_multiply_untied_forward_backward_fast_cuda_kernel<11, items_per_thread_val, 2> \
          <<<grid, block, 0, stream>>>(twiddle_cos_a, twiddle_sin_a, diagonal_a, input_reader, grad_reader, inter_storage, d_twiddle_a, d_diagonal_a, d_input_writer, batch_size, nblocks); \
        } while(0);
        // MAP(CASE_IPT_11, 1, 2, 4, 6, 8, 12, 16, 24);
        MAP(CASE_IPT_9, 6, 8, 12, 16);
        break;
    }
  });
  // Have to keep this #undef outside the AT_DISPATCH_FLOATING_TYPES macro for it to work
  #undef CASE_IPT_9
  #undef CASE_IPT_10
  #undef CASE_IPT_11
  TORCH_CHECK(cudaGetLastError() == cudaSuccess,
     "butterfly_odo_multiply_untied_forward_backward_fast_cuda_benchmark failed with error code ",
     cudaGetLastError());
}

#endif // BFLY_BENCHMARK
