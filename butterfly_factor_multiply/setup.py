from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

ext_modules = []
extension = CppExtension('butterfly_factor_multiply', ['butterfly_factor_multiply.cpp'])
ext_modules.append(extension)

setup(
    name='extension',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension})
