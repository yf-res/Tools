from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# use std=c++17 if you want to use Intel's library

ext_modules = [
    Extension(
        "nuc_wrapper",
        sources=["nuc_wrapper.pyx", "nuc.cpp"],
        include_dirs=[np.get_include()],
        language="c++",
        extra_compile_args=["-std=c++11", "-fopenmp"],
        extra_link_args=["-fopenmp"]
    )
]

setup(
    name="nuc_wrapper",
    ext_modules=cythonize(ext_modules, language_level="3"),
)