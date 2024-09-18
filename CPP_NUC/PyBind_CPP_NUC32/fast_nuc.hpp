#pragma once
#include <vector>
#include <cstdint>

class FastNUC {
public:
    FastNUC(const std::vector<float>& gain, const std::vector<float>& offset, int width, int height);
    std::vector<uint16_t> correct(const std::vector<uint16_t>& input);

private:
    std::vector<float> gain_;
    std::vector<float> offset_;
    int width_;
    int height_;
};