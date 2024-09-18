#ifndef NUC_HPP
#define NUC_HPP

#include <cstdint>

void nuc_correction(uint16_t* input, float* output, int width, int height, float* gain, float* offset);

#endif // NUC_HPP