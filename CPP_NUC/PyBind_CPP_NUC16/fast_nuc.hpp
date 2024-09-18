#pragma once
#include <vector>
#include <cstdint>

class FastNUC {
public:
    FastNUC(const std::vector<uint16_t>& gain, const std::vector<uint16_t>& offset, int width, int height);
    std::vector<uint16_t> correct(const std::vector<uint16_t>& input);

private:
    std::vector<uint16_t> gain_;
    std::vector<uint16_t> offset_;
    int width_;
    int height_;
};