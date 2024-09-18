#include "fast_nuc.hpp"
#include <iostream>
#include <chrono>
#include <random>

int main() {
    const int width = 1920;
    const int height = 1080;
    const int size = width * height;

    std::vector<float> gain(size);
    std::vector<float> offset(size);
    std::vector<uint16_t> input(size);
    std::vector<uint16_t> output(size);

    // Initialize with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> gain_dist(0.9f, 1.1f);
    std::uniform_real_distribution<> offset_dist(-100.0f, 100.0f);
    std::uniform_int_distribution<> input_dist(0, 65535);

    for (int i = 0; i < size; ++i) {
        gain[i] = gain_dist(gen);
        offset[i] = offset_dist(gen);
        input[i] = input_dist(gen);
    }

    FastNUC nuc(gain, offset, width, height);

    // Warm-up run
    nuc.correct(input, output);

    // Timed run
    const int num_runs = 100;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_runs; ++i) {
        nuc.correct(input, output);
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = end - start;
    double avg_time = diff.count() / num_runs;

    std::cout << "Average time per correction: " << avg_time * 1000 << " ms" << std::endl;
    std::cout << "Frames per second: " << 1.0 / avg_time << std::endl;

    return 0;
}