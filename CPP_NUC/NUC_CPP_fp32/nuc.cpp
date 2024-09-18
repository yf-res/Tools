// Using Intel's Threading Building Blocks (TBB) library
#include "nuc.hpp"
#include <algorithm>
#include <execution>
#include <vector>

void nuc_correction(uint16_t* input, float* output, int width, int height, float* gain, float* offset) {
    std::vector<int> indices(width * height);
    std::iota(indices.begin(), indices.end(), 0);

    std::for_each(std::execution::par_unseq, indices.begin(), indices.end(),
        [&](int i) {
            output[i] = std::max(0.0f, std::min(65535.0f, input[i] * gain[i] + offset[i]));
        }
    );
}


// Without using Intel's library
#include "nuc.hpp"
#include <algorithm>
#include <cmath>
#include <omp.h>

void nuc_correction(uint16_t* input, float* output, int width, int height, float* gain, float* offset) {
    int total_pixels = width * height;

    #pragma omp parallel for
    for (int i = 0; i < total_pixels; ++i) {
        float corrected = input[i] * gain[i] + offset[i];
        output[i] = std::max(0.0f, std::min(65535.0f, corrected));
    }
}