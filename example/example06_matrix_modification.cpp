// ==================================================================================
// Libcumat Example
// ==================================================================================
// This example is a part of the following demonstrations:
//
// 01. Instantiation
// 02. Printing
// 03. Accessing/assigning matrix elements
// 04. Matrix expressions
// 05. Matrix transpose methods
// > 06. Matrix modification methods
// 07. Matrix math methods
//
// This example is built under the "example06_matrix_modification" target when building.
// It can also be built by default by defining LIBCUMAT_BUILD_EXAMPLE when running CMake:
//
//  > cmake <libcumat-root-directory> -DLIBCUMAT_BUILD_EXAMPLES=TRUE
//  > make (or cmake --build <build-folder>)

// ==================================================================================
// 06. Matrix modification methods
// ==================================================================================
// This section will go over a few methods for efficiently modifying matrix data.
// Namely, this will cover the following methods:
//
//  (1) .resize()
//  (2) .swap()
//  (3) .fill()
//  (4) .zero()
//  (5) .copy()
//  (6) .rand()
//  (7) .identity()
//
// The methods .transpose() (matrix transpose) and .mmul() (matrix multiplication)
// are covered in sections 05 and 07 respectively.
// ==================================================================================

#include <iostream>
#include "libcumat.h"

int main(int argc, char const* argv[])
{
    // ==================================================================================
    // Matrix resize
    // ==================================================================================
    // A matrix can be resized with the .resize() method. If the new dimensions result in
    // the same size as the previous dimensions, no memory reallocation occurs and the
    // elements are rearranged in a way to fit the new dimensions. Otherwise, memory
    // reallocation occurs to expand or shrink the underlying array. This causes
    // destructive resizing, and the coefficients of the matrix may no longer be valid.
    
    std::cout << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl;
    std::cout << "06. Matrix modification methods" << std::endl;
    std::cout << "  Matrix resize" << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl << std::endl;

    // Instantiate a random float matrix
    Cumat::Matrixf mat_resize_example = Cumat::Matrixf::random(5, 6);

    std::cout << "mat_resize_example (before resize):" << std::endl << std::endl;
    std::cout << mat_resize_example << std::endl << std::endl;

    // Resize with new dimensions that still result in the same size as before
    mat_resize_example.resize(6, 5);

    std::cout << "mat_resize_example (after resize without changing resulting size):" << std::endl << std::endl;
    std::cout << mat_resize_example << std::endl << std::endl;

    // Resize with new dimensions that changes the total size
    mat_resize_example.resize(5, 4);

    std::cout << "mat_resize_example (after resize changing resulting size):" << std::endl << std::endl;
    std::cout << mat_resize_example << std::endl;

    // ==================================================================================
    // Matrix swap
    // ==================================================================================
    // Two matrices can be swapped efficiently using the .swap() method. The two matrices
    // being swapped must have the same type in order for the method to work.
    
    std::cout << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl;
    std::cout << "06. Matrix modification methods" << std::endl;
    std::cout << "  Matrix swap" << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl << std::endl;

    // Instantiate two float matrices to be swapped
    Cumat::Matrixf mat_A = Cumat::Matrixf::random(4, 6);
    Cumat::Matrixf mat_B = Cumat::Matrixf::random(6, 5);

    std::cout << "mat_A (before swap):" << std::endl << std::endl;
    std::cout << mat_A << std::endl << std::endl;
    std::cout << "mat_B (before swap):" << std::endl << std::endl;
    std::cout << mat_B << std::endl << std::endl;

    // Swap the two matrices
    mat_A.swap(mat_B);

    std::cout << "mat_A (after swap):" << std::endl << std::endl;
    std::cout << mat_A << std::endl << std::endl;
    std::cout << "mat_B (after swap):" << std::endl << std::endl;
    std::cout << mat_B << std::endl;

    // ==================================================================================
    // Matrix fill/zero
    // ==================================================================================
    // A matrix can be filled with a numeric value using the .fill() method. The .zero()
    // method is also provided for convenience for zeroing a matrix.

    // Instantiate a float matrix to be filled
    Cumat::Matrixf mat_fill_example = Cumat::Matrixf::random(5, 6);
    
    std::cout << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl;
    std::cout << "06. Matrix modification methods" << std::endl;
    std::cout << "  Matrix fill" << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl << std::endl;

    std::cout << "mat_fill_example (before fill):" << std::endl << std::endl;
    std::cout << mat_fill_example << std::endl << std::endl;

    // Fill the matrix with 4.51
    mat_fill_example.fill(4.51);

    std::cout << "mat_fill_example (after fill):" << std::endl << std::endl;
    std::cout << mat_fill_example << std::endl << std::endl;

    // Zero the matrix
    mat_fill_example.zero();

    std::cout << "mat_fill_example (after zeroing):" << std::endl << std::endl;
    std::cout << mat_fill_example << std::endl;

    // ==================================================================================
    // Matrix iterator copy
    // ==================================================================================
    // This library offers a method to copy the contents of a container into the
    // underlying vector representing the matrix. This is done with the .copy() method,
    // which accepts two InputIterator types, marking the beginning and the end of the
    // part of the container to copy.
    //
    // If the length of the part to be copied is shorter than the total size of the
    // matrix, then the coefficients are filled in a row-major order up until all items
    // are copied.
    //
    // If the length of the part to be copied is longer than the total size of the
    // matrix, then the matrix is resized to be a row vector containing all items to be
    // copied.

    std::cout << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl;
    std::cout << "06. Matrix modification methods" << std::endl;
    std::cout << "  Matrix iterator copy" << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl << std::endl;

    // Instantiate a float matrix filled with all 0
    Cumat::Matrixf mat_copy_example(5, 6);

    // Instantiate a vector with 40 elements
    std::vector<double> data(40);

    // Fill vector data with corresponding indices
    for (int i = 0; i < data.size(); ++i)
        data[i] = i;

    std::cout << "mat_copy_example (before copy):" << std::endl << std::endl;
    std::cout << mat_copy_example << std::endl << std::endl;

    // Copy only a subset of the data vector, less than the size of the matrix
    mat_copy_example.copy(data.begin(), data.begin() + 25);

    std::cout << "mat_copy_example (after copying less data than size of matrix):" << std::endl << std::endl;
    std::cout << mat_copy_example << std::endl << std::endl;

    // Copy the entire vector, greater than the size of the matrix
    mat_copy_example.copy(data.begin(), data.end());

    std::cout << "mat_copy_example (after copying more data than size of matrix):" << std::endl << std::endl;
    std::cout << mat_copy_example << std::endl;

    // ==================================================================================
    // Randomized matrix
    // ==================================================================================
    // A matrix can be randomized using the .rand() method. By default, the randomized
    // values are between -1 and 1, but the method can also take 2 arguments specifying
    // the min and max of the generated values.
    //
    // The library also provides static methods which return a randomized matrix of the
    // corresponding type:
    //
    //  Cumat::Matrixf::random(n, m)    // returns a random float matrix of size n x m
    //  Cumat::Matrixd::random(n, m)    // returns a random double matrix of size n x m

    // Instantiate a float matrix filled with 0
    Cumat::Matrixf mat_rand_example(5, 5);

    std::cout << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl;
    std::cout << "06. Matrix modification methods" << std::endl;
    std::cout << "  Randomized matrix" << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl << std::endl;

    std::cout << "mat_rand_example (before randomizing):" << std::endl << std::endl;
    std::cout << mat_rand_example << std::endl << std::endl;

    // Default randomizing: values are bounded between -1 and 1
    mat_rand_example.rand();

    std::cout << "mat_rand_example (default randomizing):" << std::endl << std::endl;
    std::cout << mat_rand_example << std::endl << std::endl;

    // Custom randomizing: values are bounded between min and max
    mat_rand_example.rand(-40, 89.4);

    std::cout << "mat_rand_example (custom randomizing):" << std::endl << std::endl;
    std::cout << mat_rand_example << std::endl;

    // ==================================================================================
    // Identity matrix
    // ==================================================================================
    // A matrix can be filled with the identity matrix by using the .identity() method.
    // This will fill each coefficient on the diagonal of the matrix starting from the
    // top left corner with 1, and all other entries will be set to 0.
    //
    // The library also provides static methods which return an identity matrix of the
    // corresponding type:
    //
    //  Cumat::Matrixf::identity(n, m)  // Returns a float identity matrix of size n x m
    //  Cumat::Matrixd::identity(n, m)  // Returns a double identity matrix of size n x m

    std::cout << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl;
    std::cout << "06. Matrix modification methods" << std::endl;
    std::cout << "  Identity matrix" << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl << std::endl;

    // Instantiate a float matrix initiated randomly
    Cumat::Matrixf mat_ident_example = Cumat::Matrixf::random(5, 5);

    std::cout << "mat_ident_example (before identity method):" << std::endl << std::endl;
    std::cout << mat_ident_example << std::endl << std::endl;

    // Set the matrix as an identity matrix
    mat_ident_example.identity();

    std::cout << "mat_ident_example (after identity method):" << std::endl << std::endl;
    std::cout << mat_ident_example << std::endl;

    return 0;
}
