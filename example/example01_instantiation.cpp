// ==================================================================================
// Libcumat Example
// ==================================================================================
// This example is a part of the following demonstrations:
//
// > 01. Instantiation
// 02. Printing
// 03. Accessing/assigning matrix elements
// 04. Matrix expressions
// 05. Matrix transpose methods
// 06. Matrix modification methods
// 07. Matrix math methods
//
// This example is built under the "example01_instantiation" target when building.
// It can also be built by default by defining LIBCUMAT_BUILD_EXAMPLE when running CMake:
//
//  > cmake <libcumat-root-directory> -DLIBCUMAT_BUILD_EXAMPLES=TRUE
//  > make (or cmake --build <build-folder>)

// ==================================================================================
// 01. Instantiation
// ==================================================================================
// This library provides both a float and a double matrix, which are instantiated
// using the following typedefs:
//
//  - Cumat::Matrixf (float matrix)
//  - Cumat::Matrixd (double matrix)
// 
// This section will cover several common types of initialization methods. For a
// complete listing of all constructor types, see libcumat_matrix.h in the folder
// include/src.
// ==================================================================================

#include <iostream>
#include "libcumat.h"

int main(int argc, char const* argv[])
{
    // ==================================================================================
    // Constructing empty matrices using default constructor
    // ==================================================================================
    // If no arguments are provided to the constructor, then by default no memory allocation
    // occurs, and the matrix is of size 0 x 0.

    // Construct a float and a double matrix using the default constructor
    Cumat::Matrixf empty_float_mat;
    Cumat::Matrixd empty_double_mat;

    std::cout << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl;
    std::cout << "01. Instantiation" << std::endl;
    std::cout << "  Constructing empty matrices using default constructor." << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl << std::endl;
    std::cout << "empty_float_mat has dimensions " << empty_float_mat.rows() << " x " << empty_float_mat.cols() << "." << std::endl;
    std::cout << "empty_double_mat has dimensions " << empty_double_mat.rows() << " x " << empty_double_mat.cols() << "." << std::endl << std::endl;

    // ==================================================================================
    // Constructing matrices using specified dimensions
    // ==================================================================================
    // A matrix can be constructed with specified dimensions and an optional fill value

    // This constructs a float matrix with 5 rows and 4 columns, with all values initialized to 0
    Cumat::Matrixf mat_with_dimensions(5, 4);

    // This constructs a float matrix with 3 rows and 6 columns, with all values initialized to 4.5
    Cumat::Matrixf initialized_mat_with_dimensions(3, 6, 4.5);

    std::cout << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl;
    std::cout << "01. Instantiation" << std::endl;
    std::cout << "  Constructing matrices using specified dimensions" << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl << std::endl;
    std::cout << "mat_with_dimensions has dimensions " << mat_with_dimensions.rows() << " x " << mat_with_dimensions.cols() << "." << std::endl << std::endl;
    std::cout << mat_with_dimensions << std::endl << std::endl;
    std::cout << "initialized_mat_with_dimensions has dimensions " << initialized_mat_with_dimensions.rows() << " x " << initialized_mat_with_dimensions.cols() << "." << std::endl << std::endl;
    std::cout << initialized_mat_with_dimensions << std::endl;

    // ==================================================================================
    // Constructing a random matrix
    // ==================================================================================
    // The library provides a static method for generating a random matrix.
    // This can be used to construct a matrix with random values using the assignment constructor.
    //
    // Note: there is no need to manually provide a seed. The generator is automatically seeded upon first use.

    // This constructs a random float matrix with 5 rows and 6 columns
    Cumat::Matrixf random_mat = Cumat::Matrixf::random(5, 6);

    std::cout << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl;
    std::cout << "01. Instantiation" << std::endl;
    std::cout << "  Constructing a random matrix" << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl << std::endl;
    std::cout << "random_mat has dimensions " << random_mat.rows() << " x " << random_mat.cols() << "." << std::endl << std::endl;
    std::cout << random_mat << std::endl;

    // ==================================================================================
    // Constructing a matrix from an expression
    // ==================================================================================
    // Matrices can be constructed from math expressions (More on expressions further down).
    // The expression is evaluated and the result is placed in the constructed matrix.

    // Here, the constructed matrix has the same dimensions as random_mat and holds the result of the written expression
    Cumat::Matrixf constructed_expression_result = random_mat * 3 + 2;

    std::cout << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl;
    std::cout << "01. Instantiation" << std::endl;
    std::cout << "  Constructing a matrix from an expression" << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl << std::endl;
    std::cout << "constructed_expression_result has dimensions " << constructed_expression_result.rows() << " x " << constructed_expression_result.cols() << "." << std::endl << std::endl;
    std::cout << constructed_expression_result << std::endl;

    // ==================================================================================
    // Constructing a matrix from input iterators
    // ==================================================================================
    // Matrices can be constructed from an existing vector using input iterators.
    // This will construct a matrix with dimensions 1 x N, where N is the length of the
    // part of the vector pointed to by the two iterators.

    // Initialize a std::vector and load it with values
    std::vector<float> v(6);

    for (size_t i = 0; i < v.size(); ++i)
        v[i] = i;

    // Construct a double matrix using this vector
    Cumat::Matrixd iterator_constructed_matrix(v.begin(), v.end());

    std::cout << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl;
    std::cout << "01. Instantiation" << std::endl;
    std::cout << "  Constructing a matrix from input iterators" << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl << std::endl;
    std::cout << "iterator_constructed_matrix has dimensions " << iterator_constructed_matrix.rows() << " x " << iterator_constructed_matrix.cols() << "." << std::endl << std::endl;
    std::cout << iterator_constructed_matrix << std::endl << std::endl;

    // The matrix can then be resized (if the total # of elements remains the same, no memory reallocation occurs)
    iterator_constructed_matrix.resize(3, 2);

    std::cout << "iterator_constructed_matrix (resized) has dimensions " << iterator_constructed_matrix.rows() << " x " << iterator_constructed_matrix.cols() << "." << std::endl << std::endl;
    std::cout << iterator_constructed_matrix << std::endl;

    // ==================================================================================
    // Constructing a matrix from a vector
    // ==================================================================================
    // Matrices can be constructed from an existing vector directly.
    // This will construct a matrix with dimensions 1 x N, where N is the length of the
    // vector.

    // Construct a float matrix using the previously defined vector
    Cumat::Matrixf vector_constructed_matrix(v);

    std::cout << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl;
    std::cout << "01. Instantiation" << std::endl;
    std::cout << "  Constructing a matrix from a vector" << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl << std::endl;
    std::cout << "vector_constructed_matrix has dimensions " << vector_constructed_matrix.rows() << " x " << vector_constructed_matrix.cols() << "." << std::endl << std::endl;
    std::cout << vector_constructed_matrix << std::endl;

	return 0;
}
