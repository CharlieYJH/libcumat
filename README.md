# libcumat
Libcumat is a GPU matrix library based on CUDA that is built with the intention of making writing matrix expressions easy. The library offers component-wise arithmetic operations, matrix multiplcations/transposes, and a variety of different math operations that are useful in common matrix expressions.

## Getting Started
To include the library in your C++ projects, simply include `libcumat.h` in the `include/` folder.

Although this is a header-only library, it does require CUDA libraries to be linked in order to compile successfully. There are two methods of including this library in your project.

### 1. Installing with CMake
This library comes with CMake configuration files to install the library to your computer. Below are instructions on how to install the files. In-source builds are restricted, so you'll have to build this in a separate build directory.

```sh
mkdir build
cd build
cmake <libcumat-root-dir>
make install (or cmake --build . --target install)
```

After installing, you can include the library in your CMake files as such:

```cmake
find_package(libcumat)
target_link_libraries(<your-target> libcumat)
```

By default, the files are put in the directory specified by `CMAKE_PREFIX_PATH`. You can also choose where to install the library by specifying a custom   `CMAKE_PREFIX_PATH`, but you'll have to define the same `CMAKE_PREFIX_PATH` when building your project so that `find_package()` can locate the library.


### 2. Manually including files
The only folder that needs to be included for the library to work is `include/`. The following CUDA libraries also need to be statically linked:

```
cudart
cuda
cublas
curand
nvrtc
```

These can be found in the 64-bit library folder in the CUDA toolkit installation directory.
