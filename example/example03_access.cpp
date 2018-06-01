// ==================================================================================
// Libcumat Example
// ==================================================================================
// This example is a part of the following demonstrations:
//
// 01. Instantiation
// 02. Printing
// > 03. Accessing/assigning matrix elements
// 04. Matrix expressions
// 05. Matrix transpose methods
// 06. Matrix modification methods
// 07. Matrix math methods
//
// This example is built under the "example03_access" target when building.
// It can also be built by default by defining LIBCUMAT_BUILD_EXAMPLE when running CMake:
//
//  > cmake <libcumat-root-directory> -DLIBCUMAT_BUILD_EXAMPLES=TRUE
//  > make (or cmake --build <build-folder>)

// ==================================================================================
// 03. Accessing/assigning matrix elements
// ==================================================================================
// There are two ways to access individual matrix elements:
//
//  (1) Accessing using row and column index: mat(r, c)
//  (2) Accessing using a linear index: mat(i)
// 
// All matrices are stored in row-major order, so a linear index i corresponds to
// row (i / total_cols) and column (i % total_cols). Similarly, row and column
// indices r and c correspond to linear index (r * total_cols + c). Matrix elements
// can be assigned using the same syntax.
//
// Note: Since the matrices are stored on the GPU, all individual accesses to
// matrix elements require memory accesses on the GPU. This can cause a lot of
// slowdown if done frequently, so if a value is reused often, it is best to save it
// in a local variable for quicker access.
// ==================================================================================

#include <iostream>
#include "libcumat.h"

int main(int argc, char const* argv[])
{
    // ==================================================================================
    // Accessing matrix elements
    // ==================================================================================
    // Accessing individual matrix elements can be done using the () operator.
    
    // First instantiate a randomly initiated float matrix of size 5 x 6.
    Cumat::Matrixf mat1 = Cumat::Matrixf::random(5, 6);

    // Now we can get the element at a specific row or column
    float mat1_row_column_element = mat1(3, 4);

    // We can also get an element using its linear index
    float mat1_linear_element = mat1(23);
    
    std::cout << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl;
    std::cout << "02. Accessing/assigning matrix elements" << std::endl;
    std::cout << "  Accessing matrix elements" << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl << std::endl;
    std::cout << "mat1: " << std::endl << std::endl;
    std::cout << mat1 << std::endl << std::endl;
    std::cout << "The element at row 3 and column 4 is: " << mat1_row_column_element << std::endl;
    std::cout << "The element at index 23 is: " << mat1_linear_element << std::endl;

    // ==================================================================================
    // Assigning matrix elements
    // ==================================================================================
    // Assigning matrix elements uses the same syntax as accessing.

    // For assignments, the syntax works naturally as you would expect
    mat1(2, 3) = 3;
    mat1(3, 4) += 4 + 5;
    mat1(2, 1) *= 2.4 / 2;
    mat1(4, 5) -= 0.3f - 3.4;
    mat1(3, 2) /= 2.0;

    // Assigning from another index
    mat1(0, 0) = mat1(4, 5);

    // Creates a double matrix with all values initiated to 1.03
    Cumat::Matrixd dmat(3, 3, 1.03);

    // Assignment of an index from another matrix
    mat1(1, 1) = dmat(2, 2);

    // Matrix elements can also be incremented/decremented using postfix and prefix operators
    ++mat1(2, 3);
    mat1(3, 4)--;

    // The same syntax can also be used in more complex expressions
    mat1(3, 3) = 4 + 3.0 * std::tanh(mat1(0, 0));
    mat1(4, 3) += 3.4 / dmat(0, 1);
    mat1(1, 3) = dmat(0, 2) + mat1(1, 2);
    
    std::cout << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl;
    std::cout << "02. Accessing/assigning matrix elements" << std::endl;
    std::cout << "  Assigning matrix elements" << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl << std::endl;
    std::cout << "mat1 (after modification): " << std::endl << std::endl;
    std::cout << mat1 << std::endl;

    return 0;
}
