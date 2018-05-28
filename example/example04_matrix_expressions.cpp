// ==================================================================================
// Libcumat Example
// ==================================================================================
// This example is a part of the following demonstrations:
//
// 01. Instantiation
// 02. Printing
// 03. Accessing/assigning matrix elements
// > 04. Matrix expressions
// 05. Matrix transpose methods
// 06. Matrix modification methods
// 07. Matrix math methods
//
// This example is built under the "example04_matrix_expressions" target when building.
// It can also be built by default by defining LIBCUMAT_BUILD_EXAMPLE when running CMake:
//
//  > cmake <libcumat-root-directory> -DLIBCUMAT_BUILD_EXAMPLES=TRUE
//  > make (or cmake --build <build-folder>)

// ==================================================================================
// 04. Matrix expressions
// ==================================================================================
// Matrix expressions can be written in a natural mathematical syntax. The library
// uses expression templates to ensure that any expression is evaluated as a whole
// to avoid the creation of unnecessary temporaries during expression evaluation.
//
//  Ex. result = sin(mat2) + 3 * mat1 - 2;
//
// All arithmetic operators (+, -, *, /) and math functions operate on expressions
// at a component wise level.
//
// The mmul() function performs matrix multiplication on two matrices.
//
//  Ex. result = mmul(mat1, mat2);    // result = mat1 x mat2
//
// Note that any expression involving matrix multiplication will cause a temporary
// matrix to be instantiated. In the example below, the mmul() function first
// creates a temporary matrix, then the expression is evaluated as a whole.
//
//  Ex. result = mat1 + mmul(2 * mat2, mat3 + 2) * 3;
//
// If one is only interested in performing a matrix multiplication with two matrices
// of the same type without any other expressions involved, a more efficient way is
// to use the mmul() class method. This is covered in more detail in example 07.
//
//  Ex. result.mmul(mat1, mat2);    // result = mat1 x mat2
//
// This will put the result of the matrix multiplication into the matrix that it's
// called from without creating a temporary matrix. This is the most efficient way
// of doing pure matrix multiplication.
//
// However, if any of the arguments to this method are general expressions, then
// the expressions will first be evaluated into a temporary matrix before the
// matrix multiplication is performed.
//
//  Ex. result.mmul(mat1 + 1, mat2);
//
// Here, mat1 + 1 is first evaluated as a temporary matrix, then the matrix
// multiplication is performed. No extra copy is made from mat2 because it is a
// trivial matrix.
//
// Finally, an expression can be evaluated without assigning to a matrix
// using the eval() method.
//
//  Ex. (1 + mat1 * mat2).eval();
//
// This creates a temporary matrix holding the result of this expression. Any
// matrix math methods can then be applied to this expression. This is covered
// more in example 07.
//
// A full list of component wise operators is listed below:
// ------------------------------------------------------------------
// Unary Operators
//  op(x), where x is a matrix
// ------------------------------------------------------------------
//
// Sign manipulation:
// - (negative sign)
// abs()
//
// Exponents:
// exp(), exp10(), exp2()
//
// Logarithms:
// log(), log1p(), log10(), log2()
//
// Powers/roots:
// square(), sqrt(), rsqrt()
// cube(), cbrt(), rcbrt()
//
// Trigonometric:
// sin(), asin(), sinh(), asinh()
// cos(), acos(), cosh(), acosh()
// tan(), atan(), tanh(), atanh()
//
// Rounding:
// ceil(), floor()
// round(), rint()
//
// Misc:
// sigmoid()
//
// Matrix op:
// transpose()
//
// ------------------------------------------------------------------
// Binary Operators
//  op(x, y), where x and/or y is a matrix or a scalar
// ------------------------------------------------------------------
// 
// Arithmetic:
// +, -, *, /
//
// Powers:
// pow()    - Converts both arguments to double
// powf()   - Converts both arguments to float
//
// Trigonometric:
// atan2()  - Converts both arguments to double
// atan2f() - Converts both arguments to float
//
// Extrema:
// max()    - Converts both arguments to double
// maxf()   - Converts both arguments to float
// min()    - Converts both arguments to double
// minf()   - Converts both arguments to float
//
// Matrix op:
// mmul()   - Matrix multiplication, both arguments must be matrices
// ==================================================================================

#include <iostream>
#include "libcumat.h"

int main(int argc, char const* argv[])
{
    // ==================================================================================
    // Component wise matrix expressions
    // ==================================================================================
    // Matrix expressions can be written in a natural mathematical manner. The basic
    // arithmetic operators all correspond to component-wise operations.

    // Instantiate 4 float matrices initialized randomly
    Cumat::Matrixf mat1 = Cumat::Matrixf::random(4, 5);
    Cumat::Matrixf mat2 = Cumat::Matrixf::random(4, 5);
    Cumat::Matrixf mat3 = Cumat::Matrixf::random(5, 4);
    Cumat::Matrixf mat4 = Cumat::Matrixf::random(4, 4);
    
    std::cout << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl;
    std::cout << "04. Matrix expressions" << std::endl;
    std::cout << "  Component wise matrix expressions" << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl << std::endl;
    std::cout << "mat1: " << std::endl << std::endl;
    std::cout << mat1 << std::endl << std::endl;
    std::cout << "mat2: " << std::endl << std::endl;
    std::cout << mat2 << std::endl << std::endl;

    // Component wise matrix expressions can be written like a math equation
    mat1 = 2.0 * mat2 + tanh(mat1);

    std::cout << "mat1 = 2.0 * mat2 + tanh(mat1): " << std::endl << std::endl;
    std::cout << mat1 << std::endl;

    // ==================================================================================
    // Component wise matrix expressions (with transpose)
    // ==================================================================================
    // transpose() can be used to transpose a matrix in an expression without evaluating
    // a temporary. Note that this doesn't change the matrix it's given.

    mat1 = -mat2 * abs(transpose(mat3)) + mat2(1, 1);
    
    std::cout << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl;
    std::cout << "04. Matrix expressions" << std::endl;
    std::cout << "  Component wise matrix expressions (with transpose)" << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl << std::endl;
    std::cout << "mat2: " << std::endl << std::endl;
    std::cout << mat2 << std::endl << std::endl;
    std::cout << "mat3: " << std::endl << std::endl;
    std::cout << mat3 << std::endl << std::endl;
    std::cout << "mat1 = -mat2 * abs(transpose(mat3)) + mat2(1, 1): " << std::endl << std::endl;
    std::cout << mat1 << std::endl;

    // ==================================================================================
    // Component wise matrix expressions (with matrix multiplication)
    // ==================================================================================
    // Matrix multiplication is performed with mmul(). Any expression involving this
    // function will produce a temporary matrix for each mmul() used. This function
    // can take both matrices or expressions as arguments.

    // mmul() is used just like any binary operator
    // Here, mat2 + 3 is first evaluated, then mat3 / 0.5f, then mmul(mat2 + 3, mat3 / 0.5f),
    // then finally the complete expression
    mat1 = sin(2 * mat4) - transpose(mmul(mat2 + 3, mat3 / 0.5f));
    
    std::cout << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl;
    std::cout << "04. Matrix expressions" << std::endl;
    std::cout << "  Component wise matrix expressions (with matrix multiplication)" << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl << std::endl;
    std::cout << "mat2: " << std::endl << std::endl;
    std::cout << mat2 << std::endl << std::endl;
    std::cout << "mat3: " << std::endl << std::endl;
    std::cout << mat3 << std::endl << std::endl;
    std::cout << "mat4: " << std::endl << std::endl;
    std::cout << mat4 << std::endl << std::endl;
    std::cout << "mat1 = sin(2 * mat4) - transpose(mmul(mat2 + 3, mat3 / 0.5f)): " << std::endl << std::endl;
    std::cout << mat1 << std::endl;

    // ==================================================================================
    // Component wise matrix expressions (with eval)
    // ==================================================================================
    // Any expression can be evaluated without assignment using the eval() method.

    // This creates a temporary matrix with the result of this expression
    (1 + -sigmoid(exp(mat2 + 2))).eval();

    // We can get the norm of this expression with the .norm() matrix method
    float expression_norm = (1 + -sigmoid(exp(mat2 + 2))).eval().norm();
    
    std::cout << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl;
    std::cout << "04. Matrix expressions" << std::endl;
    std::cout << "  Component wise matrix expressions (with eval)" << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl << std::endl;
    std::cout << "mat2: " << std::endl << std::endl;
    std::cout << mat2 << std::endl << std::endl;
    std::cout << "(1 + -sigmoid(abs(mat2 + 2))):" << std::endl << std::endl;
    std::cout << (1 + -sigmoid(abs(mat2 + 2))).eval() << std::endl << std::endl;
    std::cout << "Norm: " << expression_norm << std::endl;

    return 0;
}
