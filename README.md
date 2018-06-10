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

## Usage
### Instantiation
A float or double matrix can be instantiated by creating a matrix object with row/col parameters. By default, all elements are initiated to zero.
```cpp
Cumat::Matrixf mat(4, 5);    // Creates a 4 x 5 float matrix
Cumat::Matrixd mat(4, 5);    // Creates a 4 x 5 double matrix
```
The library also features many other ways of instantiating a matrix to suit different needs. Examples of these can be found in `example01_instantiation.cpp` in the `example/` folder. For example, a randomly initiated matrix can be instantiated as shown below.
```cpp
// Creates a 512 x 512 float matrix that is filled with random values
Cumat::Matrixf mat = Cumat::Matrixf::random(512, 512);
```
### Matrix Expressions
This library makes use of expression templates to allow matrix expressions to be evaluated efficiently. For example, we can write an expression such as the one below.
```cpp
result = mat1 + mat2 - mat3 * mat3 / exp(mat4);
```
Here, each of ```mat#``` and ```result``` are matrices/vectors. Using expression templates, the expression is evaluated as whole, rather than creating temporaries for each operation.

### Matrix Multiplication/Transpose

This library also supports matrix multiplication/transpose along with these expressions. Two matrices can be multiplied by using the ```mmul()``` function. A matrix can be transposed by using the ```transpose()``` function. We can mix these functions with matrix expressions like the example below.
```cpp
result = mmul(mat1, mat2) + 2 * transpose(mat3);
```
Note that every ```mmul()``` call necessarily produces a temporary matrix because of how matrix multiplication works. The ```*``` operator can be used for coefficient-wise multiplication. More information about matrix multiplication/transpose can be found in the ```example/``` folder.
