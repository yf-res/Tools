// fast_nuc.cpp
#include "fast_nuc.hpp"
#include <immintrin.h>
#include <omp.h>

FastNUC::FastNUC(const std::vector<uint16_t>& gain, const std::vector<uint16_t>& offset, int width, int height)
    : gain_(gain), offset_(offset), width_(width), height_(height) {}

std::vector<uint16_t> FastNUC::correct(const std::vector<uint16_t>& input) {
    std::vector<uint16_t> output(input.size());

    #pragma omp parallel for
    for (int y = 0; y < height_; ++y) {
        for (int x = 0; x < width_; x += 8) {
            __m128i gain_vec = _mm_loadu_si128((__m128i*)&gain_[y * width_ + x]);
            __m128i offset_vec = _mm_loadu_si128((__m128i*)&offset_[y * width_ + x]);
            __m128i input_vec = _mm_loadu_si128((__m128i*)&input[y * width_ + x]);

            // Multiply input by gain (16-bit x 16-bit = 32-bit)
            __m256i mul_result = _mm256_mullo_epi16(_mm256_cvtepu16_epi32(input_vec), _mm256_cvtepu16_epi32(gain_vec));

            // Add offset (32-bit + 16-bit = 32-bit)
            __m256i add_result = _mm256_add_epi32(mul_result, _mm256_cvtepu16_epi32(offset_vec));

            // Shift right by 8 to bring back to 16-bit range (equivalent to dividing by 256)
            __m256i shift_result = _mm256_srli_epi32(add_result, 8);

            // Pack the 32-bit integers into 16-bit integers
            __m128i result = _mm_packus_epi32(_mm256_extracti128_si256(shift_result, 0), _mm256_extracti128_si256(shift_result, 1));

            _mm_storeu_si128((__m128i*)&output[y * width_ + x], result);
        }
    }

    return output;
}