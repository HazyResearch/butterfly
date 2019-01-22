from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
  Extension('ABCD_mult', ['ABCD_mult.pyx'],
            include_dirs = [numpy.get_include()],
            extra_compile_args=['-O3']
  ),
]

setup(
    ext_modules = cythonize(extensions),
)
