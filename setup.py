# Adapted from https://github.com/rusty1s/pytorch_scatter/blob/master/setup.py
import os
from pathlib import Path
from setuptools import setup, find_packages

import torch
from torch.utils.cpp_extension import BuildExtension
from torch.utils.cpp_extension import CppExtension, CUDAExtension, CUDA_HOME

WITH_CUDA = torch.cuda.is_available() and CUDA_HOME is not None
if os.getenv('FORCE_CUDA', '0') == '1':
    WITH_CUDA = True
if os.getenv('FORCE_CPU', '0') == '1':
    WITH_CUDA = False

BUILD_DOCS = os.getenv('BUILD_DOCS', '0') == '1'


def get_extensions():
    Extension = CppExtension
    define_macros = []
    extra_compile_args = {'cxx': []}

    if WITH_CUDA:
        Extension = CUDAExtension
        define_macros += [('WITH_CUDA', None)]
        nvcc_flags = os.getenv('NVCC_FLAGS', '')
        nvcc_flags = [] if nvcc_flags == '' else nvcc_flags.split(' ')
        nvcc_flags += ['-arch=sm_35', '--expt-relaxed-constexpr']
        nvcc_flags += ['--expt-extended-lambda', '-lineinfo']
        extra_compile_args['nvcc'] = nvcc_flags

    extensions_dir = Path(__file__).absolute().parent / 'csrc'
    extensions = []
    for main in extensions_dir.glob('*.cpp'):
        name = main.stem
        sources = [str(main)]
        path = extensions_dir / 'cpu' / f'{name}_cpu.cpp'
        if path.exists():
            sources.append(str(path))
        path = extensions_dir / 'cuda' / f'{name}_cuda.cu'
        if WITH_CUDA and path.exists():
            sources.append(str(path))
        extension = Extension(
            'torch_butterfly._' + name,
            sources,
            include_dirs=[extensions_dir],
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
        extensions.append(extension)

    return extensions


install_requires = []
# setup_requires = ['pytest-runner']
setup_requires = []
# tests_require = ['pytest', 'pytest-cov']
tests_require = []

setup(
    name='torch_butterfly',
    version='0.0.0',
    author='Tri Dao',
    author_email='trid@stanford.edu',
    url='https://github.com/hazyresearch/learning-circuits',
    description=('Butterfly matrix multiplication in PyTorch'),
    keywords=[
        'pytorch',
        'butterfly',
        'kaleidoscope',
        'fft',
    ],
    license='Apache',
    python_requires='>=3.6',
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    ext_modules=get_extensions() if not BUILD_DOCS else [],
    cmdclass={
        'build_ext':
        BuildExtension.with_options(no_python_abi_suffix=True, use_ninja=True)
    },
    # packages=find_packages(),
    packages=['torch_butterfly'],
)
