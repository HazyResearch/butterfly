from setuptools import setup
import torch.cuda
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension
from torch.utils.cpp_extension import CUDA_HOME

# On Pytorch 1.3 the compiler (nvcc) segfaults on P100 if we don't specify the CUDA architecture (i.e. Pytorch will set it to 6.0)
ext_modules = []
if torch.cuda.is_available() and CUDA_HOME is not None:
    extension = CUDAExtension(
        'factor_multiply_fast', [
            'butterfly_multiply.cpp',
            'butterfly_multiply_cuda.cu'
        ],
        extra_compile_args={'cxx': ['-g', '-march=native', '-funroll-loops'],
                            # 'nvcc': ['-arch=sm_60', '-O2', '-lineinfo']})
                            # 'nvcc': ['-arch=sm_60', '-O2', '--expt-extended-lambda', '--expt-relaxed-constexpr',  '-lineinfo']})
                            'nvcc': ['-arch=sm_30', '-O2', '--expt-extended-lambda', '--expt-relaxed-constexpr',  '-lineinfo']
                                    + (['-arch=sm_70'] if torch.cuda.get_device_capability() == (7, 0) else [])
        })
    ext_modules.append(extension)
# extension = CppExtension('factor_multiply', ['factor_multiply.cpp'], extra_compile_args=['-march=native'])
# extension = CppExtension('factor_multiply', ['factor_multiply.cpp'])
# ext_modules.append(extension)

setup(
    name='factor_multiply_fast',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension})
