cmake_minimum_required(VERSION 3.10)
project(FastNUC)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma -fopenmp -O3 -march=native")

# Add this line to find Python
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Add these lines to find pybind11
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 CONFIG REQUIRED)

find_package(OpenMP REQUIRED)

pybind11_add_module(fast_nuc fast_nuc.cpp python_wrapper.cpp)
target_link_libraries(fast_nuc PRIVATE OpenMP::OpenMP_CXX)