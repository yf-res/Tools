pip install pybind11 numpy

mkdir build
cd build
cmake ..
make

Remember to copy the built Python module 
(fast_nuc.*.so on Linux/Mac or fast_nuc.*.pyd on Windows) 
to your Python working directory or add its location to your Python path.