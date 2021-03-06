#include "catch.hpp"
#include "util/GPUCompare.hpp"
#include "libcumat.h"
#include <cstdlib>
#include <time.h>

TEST_CASE("Float matrix assignments", "[assignment][float]")
{
    srand(time(0));

    size_t rows = ((double)std::rand() / (double)RAND_MAX) * 500 + 1;
    size_t cols = ((double)std::rand() / (double)RAND_MAX) * 500 + 1;
    size_t size = rows * cols;

    Cumat::Matrixf mat1 = Cumat::Matrixf::random(rows, cols);
    REQUIRE(mat1.rows() == rows);
    REQUIRE(mat1.cols() == cols);
    REQUIRE(mat1.size() == size);

    Cumat::Matrixf mat2 = Cumat::Matrixf::random(rows, cols);
    REQUIRE(mat2.rows() == rows);
    REQUIRE(mat2.cols() == cols);
    REQUIRE(mat2.size() == size);

    Cumat::Matrixd mat3 = Cumat::Matrixf::random(rows, cols);
    REQUIRE(mat3.rows() == rows);
    REQUIRE(mat3.cols() == cols);
    REQUIRE(mat3.size() == size);

    Cumat::Matrixf mat4 = mat1;
    REQUIRE(approxEqual(mat4, mat1));

    SECTION("Regular assignment")
    {
        mat4 = mat2;
        REQUIRE_FALSE(approxEqual(mat4, mat1));
        REQUIRE_FALSE(approxEqual(mat4, mat3));
        REQUIRE(approxEqual(mat4, mat2));

        mat3 = mat1;
        REQUIRE_FALSE(approxEqual(mat3, mat4));
        REQUIRE_FALSE(approxEqual(mat3, mat2));
        REQUIRE(approxEqual(mat3, mat1));

        mat4 = mat3;
        REQUIRE_FALSE(approxEqual(mat4, mat2));
        REQUIRE(approxEqual(mat4, mat3));
        REQUIRE(approxEqual(mat4, mat1));
    }

    SECTION("Regular assignment (different size) (same type)")
    {
        size_t new_rows;
        size_t new_cols;
        size_t new_size;

        do {
            new_rows = ((double)std::rand() / (double)RAND_MAX) * 500 + 1;
            new_cols = ((double)std::rand() / (double)RAND_MAX) * 500 + 1;
            new_size = new_rows * new_cols;
        } while (new_rows == rows || new_cols == cols);

        Cumat::Matrixf mat5 = Cumat::Matrixf::random(new_rows, new_cols);
        REQUIRE(mat5.rows() == new_rows);
        REQUIRE(mat5.cols() == new_cols);
        REQUIRE(mat5.size() == new_size);

        mat5 = mat4;
        REQUIRE(mat5.rows() == mat4.rows());
        REQUIRE(mat5.cols() == mat4.cols());
        REQUIRE(mat5.size() == mat4.size());
        REQUIRE(approxEqual(mat5, mat4));
    }

    SECTION("Regular assignment (different size) (different type)")
    {
        size_t new_rows;
        size_t new_cols;
        size_t new_size;

        do {
            new_rows = ((double)std::rand() / (double)RAND_MAX) * 500 + 1;
            new_cols = ((double)std::rand() / (double)RAND_MAX) * 500 + 1;
            new_size = new_rows * new_cols;
        } while (new_rows == rows || new_cols == cols);

        Cumat::Matrixf mat5 = Cumat::Matrixf::random(new_rows, new_cols);
        REQUIRE(mat5.rows() == new_rows);
        REQUIRE(mat5.cols() == new_cols);
        REQUIRE(mat5.size() == new_size);

        mat5 = mat3;
        REQUIRE(mat5.rows() == mat3.rows());
        REQUIRE(mat5.cols() == mat3.cols());
        REQUIRE(mat5.size() == mat3.size());
        REQUIRE(approxEqual(mat5, mat3));
    }

    SECTION("Addtion assignment")
    {
        for (size_t i = 0; i < mat4.size(); ++i)
            mat4(i) = mat1(i) + mat3(i) * 3.4 + std::exp(mat2(i) / 2.4);

        REQUIRE_FALSE(approxEqual(mat1, mat4));

        mat1 += mat3 * 3.4 + exp(mat2 / 2.4);

        REQUIRE(approxEqual(mat1, mat4));
    }

    SECTION("Subtraction assignment")
    {
        for (size_t i = 0; i < mat4.size(); ++i)
            mat4(i) = mat1(i) - (mat2(i) - mat1(i) + mat3(i) * 0.4 * mat3(i));

        REQUIRE_FALSE(approxEqual(mat1, mat4));

        mat1 -= mat2 - mat1 + mat3 * 0.4 * mat3;

        REQUIRE(approxEqual(mat1, mat4));
    }

    SECTION("Multiplication assignment")
    {
        for (size_t i = 0; i < mat4.size(); ++i)
            mat4(i) = mat1(i) * (mat2(i) + mat3(i) * mat3(i) * mat3(i) * 0.34f + mat1(i));

        REQUIRE_FALSE(approxEqual(mat1, mat4));

        mat1 *= mat2 + cube(mat3) * 0.34f + mat1;

        REQUIRE(approxEqual(mat1, mat4));
    }

    SECTION("Division assignment")
    {
        for (size_t i = 0; i < mat4.size(); ++i)
            mat4(i) = mat1(i) / (std::abs(mat3(i)) * 100.323 + std::cbrt(mat1(i)));

        REQUIRE_FALSE(approxEqual(mat1, mat4));

        mat1 /= abs(mat3) * 100.323 + cbrt(mat1);

        REQUIRE(approxEqual(mat1, mat4, 1e-3f));
    }

    SECTION("Numerical assignment")
    {
        mat1 = 3;
        REQUIRE(mat1.rows() == rows);
        REQUIRE(mat1.cols() == cols);
        REQUIRE(mat1.size() == size);
        REQUIRE(approxEqual(mat1, 3));

        mat2 = mat1(0, 0) + mat1(0, 0);
        REQUIRE(mat2.rows() == rows);
        REQUIRE(mat2.cols() == cols);
        REQUIRE(mat2.size() == size);
        REQUIRE(approxEqual(mat2, 6));
    }
}

TEST_CASE("Double matrix assignments", "[assignment][double]")
{
    size_t rows = ((double)std::rand() / (double)RAND_MAX) * 500 + 1;
    size_t cols = ((double)std::rand() / (double)RAND_MAX) * 500 + 1;
    size_t size = rows * cols;

    Cumat::Matrixd mat1 = Cumat::Matrixd::random(rows, cols);
    REQUIRE(mat1.rows() == rows);
    REQUIRE(mat1.cols() == cols);
    REQUIRE(mat1.size() == size);

    Cumat::Matrixf mat2 = Cumat::Matrixd::random(rows, cols);
    REQUIRE(mat2.rows() == rows);
    REQUIRE(mat2.cols() == cols);
    REQUIRE(mat2.size() == size);

    Cumat::Matrixd mat3 = Cumat::Matrixd::random(rows, cols);
    REQUIRE(mat3.rows() == rows);
    REQUIRE(mat3.cols() == cols);
    REQUIRE(mat3.size() == size);

    Cumat::Matrixd mat4 = mat1;
    REQUIRE(approxEqual(mat4, mat1));

    SECTION("Regular assignment")
    {
        mat4 = mat2;
        REQUIRE_FALSE(approxEqual(mat4, mat1));
        REQUIRE_FALSE(approxEqual(mat4, mat3));
        REQUIRE(approxEqual(mat4, mat2));

        mat3 = mat1;
        REQUIRE_FALSE(approxEqual(mat3, mat4));
        REQUIRE_FALSE(approxEqual(mat3, mat2));
        REQUIRE(approxEqual(mat3, mat1));

        mat4 = mat3;
        REQUIRE_FALSE(approxEqual(mat4, mat2));
        REQUIRE(approxEqual(mat4, mat3));
        REQUIRE(approxEqual(mat4, mat1));
    }

    SECTION("Regular assignment (different size) (same type)")
    {
        size_t new_rows;
        size_t new_cols;
        size_t new_size;

        do {
            new_rows = ((double)std::rand() / (double)RAND_MAX) * 500 + 1;
            new_cols = ((double)std::rand() / (double)RAND_MAX) * 500 + 1;
            new_size = new_rows * new_cols;
        } while (new_rows == rows || new_cols == cols);

        Cumat::Matrixd mat5 = Cumat::Matrixd::random(new_rows, new_cols);
        REQUIRE(mat5.rows() == new_rows);
        REQUIRE(mat5.cols() == new_cols);
        REQUIRE(mat5.size() == new_size);

        mat5 = mat4;
        REQUIRE(mat5.rows() == mat4.rows());
        REQUIRE(mat5.cols() == mat4.cols());
        REQUIRE(mat5.size() == mat4.size());
        REQUIRE(approxEqual(mat5, mat4));
    }

    SECTION("Regular assignment (different size) (different type)")
    {
        size_t new_rows;
        size_t new_cols;
        size_t new_size;

        do {
            new_rows = ((double)std::rand() / (double)RAND_MAX) * 500 + 1;
            new_cols = ((double)std::rand() / (double)RAND_MAX) * 500 + 1;
            new_size = new_rows * new_cols;
        } while (new_rows == rows || new_cols == cols);

        Cumat::Matrixd mat5 = Cumat::Matrixd::random(new_rows, new_cols);
        REQUIRE(mat5.rows() == new_rows);
        REQUIRE(mat5.cols() == new_cols);
        REQUIRE(mat5.size() == new_size);

        mat5 = mat2;
        REQUIRE(mat5.rows() == mat2.rows());
        REQUIRE(mat5.cols() == mat2.cols());
        REQUIRE(mat5.size() == mat2.size());
        REQUIRE(approxEqual(mat5, mat2));
    }

    SECTION("Addtion assignment")
    {
        for (size_t i = 0; i < mat4.size(); ++i)
            mat4(i) = mat1(i) + mat3(i) * 3.4 + std::exp(mat2(i) / 2.4);

        REQUIRE_FALSE(approxEqual(mat1, mat4));

        mat1 += mat3 * 3.4 + exp(mat2 / 2.4);

        REQUIRE(approxEqual(mat1, mat4));
    }

    SECTION("Subtraction assignment")
    {
        for (size_t i = 0; i < mat4.size(); ++i)
            mat4(i) = mat1(i) - (mat2(i) - mat1(i) + mat3(i) * 0.4 * mat3(i));

        REQUIRE_FALSE(approxEqual(mat1, mat4));

        mat1 -= mat2 - mat1 + mat3 * 0.4 * mat3;

        REQUIRE(approxEqual(mat1, mat4));
    }

    SECTION("Multiplication assignment")
    {
        for (size_t i = 0; i < mat4.size(); ++i)
            mat4(i) = mat1(i) * (mat2(i) + mat3(i) * mat3(i) * mat3(i) * 0.34f + mat1(i));

        REQUIRE_FALSE(approxEqual(mat1, mat4));

        mat1 *= mat2 + cube(mat3) * 0.34f + mat1;

        REQUIRE(approxEqual(mat1, mat4));
    }

    SECTION("Division assignment")
    {
        for (size_t i = 0; i < mat4.size(); ++i)
            mat4(i) = mat1(i) / (std::abs(mat3(i)) * 100.323 + std::cbrt(mat1(i)));

        REQUIRE_FALSE(approxEqual(mat1, mat4));

        mat1 /= abs(mat3) * 100.323 + cbrt(mat1);

        REQUIRE(approxEqual(mat1, mat4));
    }

    SECTION("Numerical assignment")
    {
        mat1 = 3.4;
        REQUIRE(mat1.rows() == rows);
        REQUIRE(mat1.cols() == cols);
        REQUIRE(mat1.size() == size);
        REQUIRE(approxEqual(mat1, 3.4));

        mat2 = mat1(0, 0) * 2;
        REQUIRE(mat2.rows() == rows);
        REQUIRE(mat2.cols() == cols);
        REQUIRE(mat2.size() == size);
        REQUIRE(approxEqual(mat2, 3.4 * 2));
    }
}
