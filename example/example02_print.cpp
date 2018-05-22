// ==================================================================================
// Libcumat Example
// ==================================================================================
// This example is a part of the following demonstrations:
//
// 01. Instantiation
// > 02. Printing
// 03. Accessing/assigning matrix elements
// 04. Matrix modification methods
// 05. Matrix transpose methods
// 06. Matrix math methods
// 07. Matrix expressions
//
// This example is built under the "example02_print" target when building.
// It can also be built by default by defining LIBCUMAT_BUILD_EXAMPLE when running CMake:
//
//  > cmake <libcumat-root-directory> -DLIBCUMAT_BUILD_EXAMPLES=TRUE
//  > make (or cmake --build <build-folder>)

// ==================================================================================
// 02. Printing
// ==================================================================================
// Printing matrices onto stdout or other streams uses standard C++ streams
// notation. This means that in order to print a certain matrix, you would print it
// like any commonly printable variable in C++ using the << operator.
// 
// General syntax: <stream-object> << mat;
//
// Note: whenever such a print occurs, assuming the matrix has N rows and
// M columns, N x M number of memory accesses occur on the GPU. This can cause a
// lot of slowdown if done often, so it's best to use prints conservatively.
// ==================================================================================

#include <iostream>
#include <fstream>
#include "libcumat.h"

int main(int argc, char const* argv[])
{
    // ==================================================================================
    // Printing matrices onto stdout
    // ==================================================================================
    // This works like other printable variables in C++ using the << operator.

    // First declare a float matrix and a double matrix initiated randomly
    Cumat::Matrixf float_mat = Cumat::Matrixf::random(5, 6);
    Cumat::Matrixd double_mat = Cumat::Matrixd::random(6, 4);
    
    std::cout << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl;
    std::cout << "02. Printing" << std::endl;
    std::cout << "  Printing matrices onto stdout" << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl << std::endl;
    std::cout << "float_mat: " << std::endl << std::endl;

    // Print syntax works like any other variable in C++
    std::cout << float_mat << std::endl << std::endl;

    std::cout << "double_mat: " << std::endl << std::endl;
    std::cout << double_mat << std::endl;

    // ==================================================================================
    // Printing matrices into file
    // ==================================================================================
    // This works similarly to printing onto stdout, but the stream is replaced with
    // a std::ofstream object instead.

    std::cout << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl;
    std::cout << "02. Printing" << std::endl;
    std::cout << "  Printing matrices into file" << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl << std::endl;

    // Make an ofstream object first
    std::ofstream out("example02_print.txt");

    // Printing to file works in the same way
    if (out.is_open()) {

        out << "float_mat: " << std::endl << std::endl;
        out << float_mat << std::endl << std::endl;

        out << "double_mat: " << std::endl << std::endl;
        out << double_mat << std::endl << std::endl;

        out.close();

        std::cout << "Matrices printed to example02_print.txt." << std::endl;
    }

    return 0;
}
