#ifndef NUC_HPP
#define NUC_HPP

#include <cstdint>

void nuc_correction(uint16_t* input, uint16_t* output, int width, int height, uint16_t* gain, uint16_t* offset);

#endif // NUC_HPP