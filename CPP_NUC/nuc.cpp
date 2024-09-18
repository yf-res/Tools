#include "nuc.hpp"
#include <algorithm>
#include <cmath>
#include <omp.h>

void nuc_correction(uint16_t* input, uint16_t* output, int width, int height, uint16_t* gain, uint16_t* offset) {
    int total_pixels = width * height;

    #pragma omp parallel for
    for (int i = 0; i < total_pixels; ++i) {
        output[i] = input[i] * gain[i] + offset[i];
    }
}
