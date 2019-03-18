#!python
#cython: boundscheck=False, wraparound=False, initializedcheck=False
import numpy as np

# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).
cimport numpy as np

# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
DTYPE = np.float32
# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.float32_t DTYPE_t
# ctypedef np.complex64_t complex_t

ctypedef fused numeric:
    float
    double
    float complex
    double complex

def ABCD_mult(np.ndarray[DTYPE_t, ndim=3] ABCD, np.ndarray[DTYPE_t, ndim=3] input, np.ndarray[DTYPE_t, ndim=3] out):
    """
    Parameters:
        ABCD: (2, 2, n)
        input: (batch_size, 2, n)
        out: (batch_size, 2, n)
    """
    cdef unsigned int batch_size = input.shape[0],  n = input.shape[2]
    cdef unsigned int b, i
    for b in range(batch_size):
        for i in range(n):
            out[b, 0, i] = ABCD[0, 0, i] * input[b, 0, i] + ABCD[0, 1, i] * input[b, 1, i]
            out[b, 1, i] = ABCD[1, 0, i] * input[b, 0, i] + ABCD[1, 1, i] * input[b, 1, i]


def ABCD_mult_inplace(np.ndarray[DTYPE_t, ndim=3] ABCD, np.ndarray[DTYPE_t, ndim=3] input):
    """
    Parameters:
        ABCD: (2, 2, n)
        input: (batch_size, 2, n)
    """
    cdef unsigned int batch_size = input.shape[0],  n = input.shape[2]
    cdef unsigned int b, i
    cdef DTYPE_t temp
    for b in range(batch_size):
        for i in range(n):
            temp = ABCD[0, 0, i] * input[b, 0, i] + ABCD[0, 1, i] * input[b, 1, i]
            input[b, 1, i] = ABCD[1, 0, i] * input[b, 0, i] + ABCD[1, 1, i] * input[b, 1, i]
            input[b, 0, i] = temp


cpdef ABCD_mult_inplace_memview(DTYPE_t[:, :, ::1] ABCD, DTYPE_t[:, :, ::1] input):
    """
    Parameters:
        ABCD: (2, 2, n)
        input: (batch_size, 2, n)
    """
    cdef unsigned int batch_size = input.shape[0],  n = input.shape[2]
    cdef unsigned int b, i
    cdef DTYPE_t temp
    for b in range(batch_size):
        for i in range(n):
            temp = ABCD[0, 0, i] * input[b, 0, i] + ABCD[0, 1, i] * input[b, 1, i]
            input[b, 1, i] = ABCD[1, 0, i] * input[b, 0, i] + ABCD[1, 1, i] * input[b, 1, i]
            input[b, 0, i] = temp


def ABCD_mult_inplace_complex(np.ndarray[np.complex64_t, ndim=3] ABCD, np.ndarray[np.complex64_t, ndim=3] input):
    """
    Parameters:
        ABCD: (2, 2, n)
        input: (batch_size, 2, n)
    """
    cdef unsigned int batch_size = input.shape[0],  n = input.shape[2]
    cdef unsigned int b, i
    cdef np.complex64_t temp
    for b in range(batch_size):
        for i in range(n):
            temp = ABCD[0, 0, i] * input[b, 0, i] + ABCD[0, 1, i] * input[b, 1, i]
            input[b, 1, i] = ABCD[1, 0, i] * input[b, 0, i] + ABCD[1, 1, i] * input[b, 1, i]
            input[b, 0, i] = temp


def ABCD_mult_inplace_generic(numeric[:, :, :] ABCD, numeric[:, :, :] input):
    """
    Parameters:
        ABCD: (2, 2, n)
        input: (batch_size, 2, n)
    """
    cdef unsigned int batch_size = input.shape[0],  n = input.shape[2]
    cdef unsigned int b, i
    cdef numeric temp
    for b in range(batch_size):
        for i in range(n):
            temp = ABCD[0, 0, i] * input[b, 0, i] + ABCD[0, 1, i] * input[b, 1, i]
            input[b, 1, i] = ABCD[1, 0, i] * input[b, 0, i] + ABCD[1, 1, i] * input[b, 1, i]
            input[b, 0, i] = temp
