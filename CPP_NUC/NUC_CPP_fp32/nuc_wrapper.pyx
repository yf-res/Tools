# distutils: language = c++
# distutils: sources = nuc.cpp

import numpy as np
cimport numpy as np
from libcpp cimport bool

cdef extern from "nuc.hpp":
    void nuc_correction(unsigned short* input, float* output, int width, int height, float* gain, float* offset)

def py_nuc_correction(np.ndarray[np.uint16_t, ndim=2, mode="c"] input not None,
                      np.ndarray[np.float32_t, ndim=2, mode="c"] gain not None,
                      np.ndarray[np.float32_t, ndim=2, mode="c"] offset not None):
    cdef int height = input.shape[0]
    cdef int width = input.shape[1]
    cdef np.ndarray[np.float32_t, ndim=2, mode="c"] output = np.empty((height, width), dtype=np.float32)

    nuc_correction(&input[0, 0], &output[0, 0], width, height, &gain[0, 0], &offset[0, 0])

    return output