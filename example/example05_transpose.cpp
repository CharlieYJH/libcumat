// ==================================================================================
// Libcumat Example
// ==================================================================================
// This example is a part of the following demonstrations:
//
// 01. Instantiation
// 02. Printing
// 03. Accessing/assigning matrix elements
// 04. Matrix expressions
// > 05. Matrix transpose methods
// 06. Matrix modification methods
// 07. Matrix math methods
//
// This example is built under the "example05_transpose" target when building.
// It can also be built by default by defining LIBCUMAT_BUILD_EXAMPLE when running CMake:
//
//  > cmake <libcumat-root-directory> -DLIBCUMAT_BUILD_EXAMPLES=TRUE
//  > make (or cmake --build <build-folder>)

// ==================================================================================
// 05. Matrix transpose methods
// ==================================================================================
// This lbrary offers three methods of matrix transpose.
//
//  (1) transpose()          - expression operator
//  (2) .transpose()         - in-place transpose
//  (3) .transpose(matrix)   - out-of-place transpose
//
// -- Option (1):
//
// The expression operator can be used when writing a matrix equation that requires
// a transpose. This does not create a temporary matrix, and is evaluated along
// with the entire matrix expression. For more on matrix expressions, see example 04.
//
//  Ex. mat1 = mat2 + transpose(mat3);
//
// Warning: Do not use the transpose operator directly if the matrix being assigned
// to is also contained within the operator. This produces unexpected behaviour
// because of an issue known as aliasing. To resolve this, use the .eval() method
// to produce a temporary transposed matrix.
//
//  Ex. mat1 = mat2 + transpose(mat1 + 1);          // Incorrect results
//      mat1 = mat2 + transpose(mat1 + 1).eval();   // Resolves aliasing issues
//
// -- Option (2):
//
// A matrix can be transposed in-place by using the .transpose() method. This is
// the most costly way of transposing, as it implicitly creates a temporary matrix
// first. This method is included for convenience, though one should try and use
// the out-of-place transpose for maximum performance.
//
//  Ex. mat1.transpose();   // mat1 is now transposed in-place
//
// -- Option (3):
//
// This option offers the best performance. A matrix can be transposed out-of-place
// using the overloaded .transpose() method. If the resulting matrix is not
// the same size as the transposed matrix, then a resize is automatically performed
// to fit the transposed matrix.
//
//  Ex. mat1.transpose(mat2);   // mat1 contains a transposed mat2
//
// Note: This method will throw an error if the matrix being transposed is the
// same as the one calling the method.
//
// An expression can also be put in the input argument, though this will create
// a temporary matrix. In this circumstance, it is best to use option (1).
// ==================================================================================

#include <iostream>
#include "libcumat.h"

int main(int argc, char const* argv[])
{
    // Instantiate float matrices initialized randomly
    Cumat::Matrixf mat1 = Cumat::Matrixf::random(4, 5);
    Cumat::Matrixf mat2 = Cumat::Matrixf::random(5, 4);

    // ==================================================================================
    // Expression transpose
    // ==================================================================================
    // transpose() can be used anywhere in a matrix expression.

    // No temporary is created for (2 * mat2), the expression is evaluated as a whole
    mat1 = 1.2 + transpose(2 * mat2);
    
    std::cout << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl;
    std::cout << "05. Matrix transpose methods" << std::endl;
    std::cout << "  Expression transpose" << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl << std::endl;
    std::cout << "mat2: " << std::endl << std::endl;
    std::cout << mat2 << std::endl << std::endl;
    std::cout << "mat1 = 1.2 + transpose(2 * mat2): " << std::endl << std::endl;
    std::cout << mat1 << std::endl;

    // ==================================================================================
    // Expression transpose (aliasing)
    // ==================================================================================
    // Unexpected results occur when transpose() contains a matrix that is being assigned
    // to. Use .eval() to resolve this issue.
    
    std::cout << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl;
    std::cout << "05. Matrix transpose methods" << std::endl;
    std::cout << "  Expression transpose (aliasing)" << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl << std::endl;

    // Create two of the same matrices initialized randomly
    Cumat::Matrixf original_matrix = Cumat::Matrixf::random(50, 50);
    Cumat::Matrixf aliased_matrix = original_matrix;
    Cumat::Matrixf unaliased_matrix = original_matrix;

    // Aliasing example
    aliased_matrix = 2 + transpose(aliased_matrix * 3);

    // Fixed example using .eval()
    unaliased_matrix = 2 + transpose(unaliased_matrix * 3).eval();

    std::cout << "Check whether the two transposes are the same:" << std::endl << std::endl;

    for (size_t i = 0; i < aliased_matrix.rows(); ++i) {
        for (size_t j = 0; j < aliased_matrix.cols(); ++j) {

            float aliased = aliased_matrix(i, j);
            float unaliased = unaliased_matrix(i, j);
            float answer = 2 + original_matrix(j, i) * 3;

            if (std::abs(aliased - answer) > 1e-4f) {
                std::cout << "Difference at index (" << i << ", " << j << ")." << std::endl;
                std::cout << "Correct answer  : " << answer << std::endl;
                std::cout << "aliased_matrix  : " << aliased << std::endl;
                std::cout << "unaliased_matrix: " << unaliased << std::endl << std::endl;
            }
        }
    }

    // ==================================================================================
    // In-place transpose
    // ==================================================================================
    // Any matrix can be transposed in-place using the .transpose() method.

    std::cout << "------------------------------------------------------------------" << std::endl;
    std::cout << "05. Matrix transpose methods" << std::endl;
    std::cout << "  In-place transpose" << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl << std::endl;

    Cumat::Matrixf mat3 = Cumat::Matrixf::random(5, 6);

    std::cout << "mat3 (before transpose): " << std::endl << std::endl;
    std::cout << mat3 << std::endl << std::endl;

    // In-place transpose using matrix class method
    mat3.transpose();

    std::cout << "mat3 (after transpose): " << std::endl << std::endl;
    std::cout << mat3 << std::endl;
 
    // ==================================================================================
    // Out-of-place transpose
    // ==================================================================================
    // Use the overlaoded .transpose() method to do an out-of-place transpose. This is
    // the most efficient transpose method.

    Cumat::Matrixf mat_A = Cumat::Matrixf::random(5, 5);
    Cumat::Matrixf mat_B = Cumat::Matrixf::random(5, 6);

    std::cout << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl;
    std::cout << "05. Matrix transpose methods" << std::endl;
    std::cout << "  In-place transpose" << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl << std::endl;

    std::cout << "mat_A (before transpose call):" << std::endl << std::endl;
    std::cout << mat_A << std::endl << std::endl;
    std::cout << "mat_B (before transpose call):" << std::endl << std::endl;
    std::cout << mat_B << std::endl << std::endl;

    // mat_A contains the transpose of mat_B after this call
    mat_A.transpose(mat_B);

    std::cout << "mat_A (after transpose call):" << std::endl << std::endl;
    std::cout << mat_A << std::endl << std::endl;
    std::cout << "mat_B (after transpose call):" << std::endl << std::endl;
    std::cout << mat_B << std::endl;

    return 0;
}
