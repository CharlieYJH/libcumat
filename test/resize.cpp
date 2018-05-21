#include "catch.hpp"
#include "util/GPUCompare.hpp"
#include "libcumat.h"

TEST_CASE("Float matrix resize", "[resize][float]")
{
    size_t mat1_rows = 250;
    size_t mat1_cols = 250;
    size_t mat1_size = mat1_rows * mat1_cols;

    Cumat::Matrixf mat1 = Cumat::Matrixf::random(mat1_rows, mat1_cols);
    REQUIRE(mat1.rows() == mat1_rows);
    REQUIRE(mat1.cols() == mat1_cols);
    REQUIRE(mat1.size() == mat1_size);

    SECTION("Resizing to a bigger size")
    {
        size_t new_rows = 3000;
        size_t new_cols = 2050;
        size_t new_size = new_rows * new_cols;

        mat1.resize(new_rows, new_cols);
        REQUIRE(mat1.rows() == new_rows);
        REQUIRE(mat1.cols() == new_cols);
        REQUIRE(mat1.size() == new_size);
        REQUIRE(mat1.device_vector().size() == new_size);
    }

    SECTION("Resizing to a smaller size")
    {
        size_t new_rows = 14;
        size_t new_cols = 10;
        size_t new_size = new_rows * new_cols;

        mat1.resize(new_rows, new_cols);
        REQUIRE(mat1.rows() == new_rows);
        REQUIRE(mat1.cols() == new_cols);
        REQUIRE(mat1.size() == new_size);
        REQUIRE(mat1.device_vector().size() == new_size);
    }

    SECTION("Resizing to the same size")
    {
        size_t new_rows = mat1_rows;
        size_t new_cols = mat1_cols;
        size_t new_size = new_rows * new_cols;

        mat1.resize(new_rows, new_cols);
        REQUIRE(mat1.rows() == new_rows);
        REQUIRE(mat1.cols() == new_cols);
        REQUIRE(mat1.size() == new_size);
        REQUIRE(mat1.device_vector().size() == new_size);
    }

    SECTION("Resizing to 0, only row parameter is 0")
    {
        size_t new_rows = 0;
        size_t new_cols = mat1_cols;
        size_t new_size = new_rows * new_cols;

        mat1.resize(new_rows, new_cols);
        REQUIRE(mat1.rows() == 0);
        REQUIRE(mat1.cols() == 0);
        REQUIRE(mat1.size() == new_size);
        REQUIRE(mat1.device_vector().size() == new_size);
    }

    SECTION("Resizing to 0, only col parameter is 0")
    {
        size_t new_rows = mat1_rows;
        size_t new_cols = 0;
        size_t new_size = new_rows * new_cols;

        mat1.resize(new_rows, new_cols);
        REQUIRE(mat1.rows() == 0);
        REQUIRE(mat1.cols() == 0);
        REQUIRE(mat1.size() == new_size);
        REQUIRE(mat1.device_vector().size() == new_size);
    }

    SECTION("Resizing to 0, both parameters are 0")
    {
        size_t new_rows = 0;
        size_t new_cols = 0;
        size_t new_size = new_rows * new_cols;

        mat1.resize(new_rows, new_cols);
        REQUIRE(mat1.rows() == 0);
        REQUIRE(mat1.cols() == 0);
        REQUIRE(mat1.size() == new_size);
        REQUIRE(mat1.device_vector().size() == new_size);
    }
}

TEST_CASE("Double matrix resize", "[resize][double]")
{
    size_t mat1_rows = 250;
    size_t mat1_cols = 250;
    size_t mat1_size = mat1_rows * mat1_cols;

    Cumat::Matrixd mat1 = Cumat::Matrixd::random(mat1_rows, mat1_cols);
    REQUIRE(mat1.rows() == mat1_rows);
    REQUIRE(mat1.cols() == mat1_cols);
    REQUIRE(mat1.size() == mat1_size);

    SECTION("Resizing to a bigger size")
    {
        size_t new_rows = 3000;
        size_t new_cols = 2050;
        size_t new_size = new_rows * new_cols;

        mat1.resize(new_rows, new_cols);
        REQUIRE(mat1.rows() == new_rows);
        REQUIRE(mat1.cols() == new_cols);
        REQUIRE(mat1.size() == new_size);
        REQUIRE(mat1.device_vector().size() == new_size);
    }

    SECTION("Resizing to a smaller size")
    {
        size_t new_rows = 14;
        size_t new_cols = 10;
        size_t new_size = new_rows * new_cols;

        mat1.resize(new_rows, new_cols);
        REQUIRE(mat1.rows() == new_rows);
        REQUIRE(mat1.cols() == new_cols);
        REQUIRE(mat1.size() == new_size);
        REQUIRE(mat1.device_vector().size() == new_size);
    }

    SECTION("Resizing to the same size")
    {
        size_t new_rows = mat1_rows;
        size_t new_cols = mat1_cols;
        size_t new_size = new_rows * new_cols;

        mat1.resize(new_rows, new_cols);
        REQUIRE(mat1.rows() == new_rows);
        REQUIRE(mat1.cols() == new_cols);
        REQUIRE(mat1.size() == new_size);
        REQUIRE(mat1.device_vector().size() == new_size);
    }

    SECTION("Resizing to 0, only row parameter is 0")
    {
        size_t new_rows = 0;
        size_t new_cols = mat1_cols;
        size_t new_size = new_rows * new_cols;

        mat1.resize(new_rows, new_cols);
        REQUIRE(mat1.rows() == 0);
        REQUIRE(mat1.cols() == 0);
        REQUIRE(mat1.size() == new_size);
        REQUIRE(mat1.device_vector().size() == new_size);
    }

    SECTION("Resizing to 0, only col parameter is 0")
    {
        size_t new_rows = mat1_rows;
        size_t new_cols = 0;
        size_t new_size = new_rows * new_cols;

        mat1.resize(new_rows, new_cols);
        REQUIRE(mat1.rows() == 0);
        REQUIRE(mat1.cols() == 0);
        REQUIRE(mat1.size() == new_size);
        REQUIRE(mat1.device_vector().size() == new_size);
    }

    SECTION("Resizing to 0, both parameters are 0")
    {
        size_t new_rows = 0;
        size_t new_cols = 0;
        size_t new_size = new_rows * new_cols;

        mat1.resize(new_rows, new_cols);
        REQUIRE(mat1.rows() == 0);
        REQUIRE(mat1.cols() == 0);
        REQUIRE(mat1.size() == new_size);
        REQUIRE(mat1.device_vector().size() == new_size);
    }
}
