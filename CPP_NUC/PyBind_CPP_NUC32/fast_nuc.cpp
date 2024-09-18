#include "fast_nuc.hpp"
#include <immintrin.h>
#include <omp.h>

FastNUC::FastNUC(const std::vector<float>& gain, const std::vector<float>& offset, int width, int height)
    : gain_(gain), offset_(offset), width_(width), height_(height) {}

std::vector<uint16_t> FastNUC::correct(const std::vector<uint16_t>& input) {
    std::vector<uint16_t> output(input.size());

    #pragma omp parallel for
    for (int y = 0; y < height_; ++y) {
        for (int x = 0; x < width_; x += 8) {
            __m256 gain_vec = _mm256_loadu_ps(&gain_[y * width_ + x]);
            __m256 offset_vec = _mm256_loadu_ps(&offset_[y * width_ + x]);
            __m256i input_vec = _mm256_cvtepu16_epi32(_mm_loadu_si128((__m128i*)&input[y * width_ + x]));
            __m256 input_float = _mm256_cvtepi32_ps(input_vec);

            __m256 result = _mm256_fmadd_ps(input_float, gain_vec, offset_vec);
            __m256i result_int = _mm256_cvtps_epi32(result);
            __m128i result_16 = _mm256_extracti128_si256(_mm256_packus_epi32(result_int, result_int), 0);

            _mm_storeu_si128((__m128i*)&output[y * width_ + x], result_16);
        }
    }

    return output;
}