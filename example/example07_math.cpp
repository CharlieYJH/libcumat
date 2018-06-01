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
// 06. Matrix modification methods
// > 07. Matrix math methods
//
// This example is built under the "example07_math" target when building.
// It can also be built by default by defining LIBCUMAT_BUILD_EXAMPLE when running CMake:
//
//  > cmake <libcumat-root-directory> -DLIBCUMAT_BUILD_EXAMPLES=TRUE
//  > make (or cmake --build <build-folder>)

// ==================================================================================
// 06. Matrix math methods
// ==================================================================================
// This library features some useful math methods for matrix related calculations.
// This example will cover the following math methods:
//
//  (1) Matrix dimensions
//  (2) .sum()
//  (3) .norm()
//  (4) .maxElement() / .minElement()
//  (5) .maxIndex() / .minIndex()
//  (6) .mmul()
//  (7) Component-wise math operations
//
// The .transpose() method is covered in detail in example 05.
// ==================================================================================

#include <iostream>
#include "libcumat.h"

int main(int argc, char const* argv[])
{
    // ==================================================================================
    // Matrix dimensions
    // ==================================================================================
    // The dimensions of a matrix can be accessed using the three methods below:
    //
    //  (1) .rows()     - Returns the total number of rows of the matrix
    //  (2) .cols()     - Returns the total number of columns of the matrix
    //  (3) .size()     - Returns the total size of the matrix (rows x cols)
    // 
    // All three methods return types of size_t.

    // Instantiate a randomly initiated float matrix
    Cumat::Matrixf mat1 = Cumat::Matrixf::random(5, 6);

    // Get the dimensions associated with this matrix
    size_t mat1_rows = mat1.rows();
    size_t mat1_cols = mat1.cols();
    size_t mat1_size = mat1.size();
    
    std::cout << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl;
    std::cout << "07. Matrix math methods" << std::endl;
    std::cout << "  Matrix dimensions" << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl << std::endl;
    std::cout << "mat1: " << std::endl << std::endl;
    std::cout << mat1 << std::endl << std::endl;
    std::cout << "mat1 (rows): " << mat1_rows << std::endl;
    std::cout << "mat1 (cols): " << mat1_cols << std::endl;
    std::cout << "mat1 (size): " << mat1_size << std::endl;

    // ==================================================================================
    // Matrix sum/norm
    // ==================================================================================
    // The sum of all the coefficients of a matrix can be found by using the .sum()
    // method. The 2-norm of a matrix (Frobenius norm) can be found by using the .norm()
    // method.

    // Get the sum of all coefficients in the matrix
    float mat1_sum = mat1.sum();

    // Get the 2-norm of the matrix
    float mat1_norm = mat1.norm();
    
    std::cout << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl;
    std::cout << "07. Matrix math methods" << std::endl;
    std::cout << "  Matrix sum/norm" << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl << std::endl;
    std::cout << "mat1: " << std::endl << std::endl;
    std::cout << mat1 << std::endl << std::endl;
    std::cout << "mat1 (sum) : " << mat1_sum << std::endl;
    std::cout << "mat1 (norm): " << mat1_norm << std::endl;

    // ==================================================================================
    // Matrix extrema
    // ==================================================================================
    // This library provides 4 methods related to matrix extrema:
    //
    //  (1) .maxElement()   - Returns the max element in the matrix
    //  (2) .minElement()   - Returns the min element in the matrix
    //  (3) .maxIndex()     - Returns the index of the max element
    //  (4) .minIndex()     - Returns the index of the min element
    //
    // If there are any repeated elements, these methods return the first instance of
    // either the max or the min. The indices correspond to the linear index of the
    // underlying array stored in row-major order. To convert it to row/col index,
    // use the following formulas:
    //  
    //  row = index / col
    //  col = index % col
    //
    // The return type of the index is size_t, so be careful if converting it to other
    // integer types.

    // Get the max and min elements of mat1
    float mat1_max = mat1.maxElement();
    float mat1_min = mat1.minElement();

    // Get the linear index of the max and min elements of mat1
    size_t mat1_max_index = mat1.maxIndex();
    size_t mat1_min_index = mat1.minIndex();
    
    std::cout << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl;
    std::cout << "07. Matrix math methods" << std::endl;
    std::cout << "  Matrix extrema" << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl << std::endl;
    std::cout << "mat1: " << std::endl << std::endl;
    std::cout << mat1 << std::endl << std::endl;
    std::cout << "mat1 (max element): " << mat1_max << std::endl;
    std::cout << "mat1 (min element): " << mat1_min << std::endl << std::endl;
    std::cout << "mat1 (max element index): " << mat1_max_index << " (row: " << mat1_max_index / mat1.cols() << ", col: " << mat1_max_index % mat1.cols() << ")" << std::endl;
    std::cout << "mat1 (min element index): " << mat1_min_index << " (row: " << mat1_min_index / mat1.cols() << ", col: " << mat1_min_index % mat1.cols() << ")" << std::endl;

    // ==================================================================================
    // Matrix multiplication
    // ==================================================================================
    // The .mmul() method takes 2 arguments, which can be either matrices or expressions.
    // The resulting matrix from the multiplication of these two matrices is placed into
    // the matrix from which the method is called.
    //
    //  Ex. mat1.mmul(mat2, mat3);      // mat1 = mat2 x mat3
    //
    // The method takes an optional third argument, which is a multiplier for mat1. It
    // works like this:
    //
    //  Ex. mat1.mmul(mat2, mat3, 2);   // mat1 = mat2 x mat3 + 2 * mat1;
    //
    // This is an efficient way to add onto mat1 if necessary, because the operation is
    // wrapped around the CUDA cuBLAS gemm operation. By default, this third argument
    // is 0.
    //
    // Note: ensure that mat2 x mat3 results in the same dimensions as mat1 if specifying
    // a non-zero third argument. Otherwise, mat1 will be resized destructively and the
    // result may not be what you expect.
    //
    // For more on matrix multiplication, check out example 04.

    // Declare two float matrices
    Cumat::Matrixf mat2 = Cumat::Matrixf::random(5, 5);
    Cumat::Matrixf mat3 = Cumat::Matrixf::random(5, 6);
    
    std::cout << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl;
    std::cout << "07. Matrix math methods" << std::endl;
    std::cout << "  Matrix multiplication" << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl << std::endl;
    std::cout << "mat1: " << std::endl << std::endl;
    std::cout << mat1 << std::endl << std::endl;
    std::cout << "mat2: " << std::endl << std::endl;
    std::cout << mat2 << std::endl << std::endl;
    std::cout << "mat3: " << std::endl << std::endl;
    std::cout << mat3 << std::endl << std::endl;

    // mat1 = mat2 x mat3 + 2 * mat1
    mat1.mmul(mat2, mat3, 2);

    std::cout << "mat1 (after multiplication): " << std::endl << std::endl;
    std::cout << mat1 << std::endl << std::endl;

    // ==================================================================================
    // Component-wise math operations
    // ==================================================================================
    // In addition to the math operations used in expressions, this library also offers
    // some math methods that can be called on matrices for convenience reasons.
    //
    //  Ex. mat1.tanh();    // mat1 = tanh(mat1);
    //
    // These operations can also be chained, but this is not advised because it is not
    // as efficient as writing a matrix expression.
    //
    //  Ex. mat1.square().tanh();   // Write mat1 = tanh(square(mat)) instead
    // 
    // Below is a full list of the available math methods:
    //
    //  abs()           - absolute value
    //  inverse()       - reciprocal
    //  clip(min, max)  - bound(mat) s.t. min <= mat <= max
    //  exp()           - exponentiation with e as base
    //  exp10()         - exponentiation with 10 as base
    //  exp2()          - exponentiation with 2 as base
    //  log()           - natural logarithm
    //  log10()         - base 10 logarithm
    //  log2()          - base 2 logarithm
    //  pow(num)        - raise matrix component to the power of num as a double
    //  powf(num)       - raise matrix component to the power of num as a float
    //  square()        - square
    //  sqrt()          - square root
    //  rsqrt()         - reciprocal square root
    //  cube()          - cube
    //  cbrt()          - cube root
    //  rcbrt()         - reciprocal cube root
    //  sin()           - sine
    //  cos()           - cosine
    //  tan()           - tangent
    //  asin()          - arcsine
    //  acos()          - arccosine
    //  atan()          - arctangent
    //  sinh()          - hyperbolic sine
    //  cosh()          - hyperbolic cosine
    //  tanh()          - hyperbolic tangent
    //  asinh()         - inverse hyperbolic sine
    //  acosh()         - inverse hyperbolic cosine
    //  atanh()         - inverse hyperbolic tangent
    //  ceil()          - ceiling function (round up)
    //  floor()         - floor function (round down)
    //  rint()          - rounding (halfway cases rounded to nearest even integer)
    //  round()         - rounding (halfway cases rounded away from 0)
    //  sigmoid()       - sigmoid function
    
    std::cout << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl;
    std::cout << "07. Matrix math methods" << std::endl;
    std::cout << "  Component-wise math operations" << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl << std::endl;
    std::cout << "mat1: " << std::endl << std::endl;
    std::cout << mat1 << std::endl << std::endl;

    // Clip values of mat1 between -1 and 0.5
    mat1.clip(-1, 0.5);

    std::cout << "mat1 (after tanh): " << std::endl << std::endl;
    std::cout << mat1 << std::endl << std::endl;

    return 0;
}
