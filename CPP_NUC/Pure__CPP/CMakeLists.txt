cmake_minimum_required(VERSION 3.10)
project(FastNUC)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma -fopenmp -O3 -march=native")

add_library(fast_nuc STATIC fast_nuc.cpp)
target_include_directories(fast_nuc PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(nuc_test main.cpp)
target_link_libraries(nuc_test PRIVATE fast_nuc)