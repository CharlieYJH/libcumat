#include "catch.hpp"
#include "util/GPUCompare.hpp"
#include "libcumat.h"

TEST_CASE("Float identity matrix", "[identity][float]")
{
	Cumat::init();

	SECTION("Identity matrix assignment")
	{
		size_t mat1_rows = 1045;
		size_t mat1_cols = 500;
		size_t mat1_size = mat1_rows * mat1_cols;

		Cumat::Matrixf mat1 = Cumat::Matrixf::identity(mat1_rows, mat1_cols);
		REQUIRE(mat1.rows() == mat1_rows);
		REQUIRE(mat1.cols() == mat1_cols);
		REQUIRE(mat1.size() == mat1_size);

		for (size_t i = 0; i < mat1.rows(); ++i) {
			for (size_t j = 0; j < mat1.cols(); ++j) {
				if (i == j)
					REQUIRE(mat1(i, j) == Approx(1.0f));
				else
					REQUIRE(mat1(i, j) == Approx(0.0f));
			}
		}
	}

	SECTION("Square identity matrix fill")
	{
		size_t mat1_rows = 1024;
		size_t mat1_cols = 1024;
		size_t mat1_size = mat1_rows * mat1_cols;

		Cumat::Matrixf mat1(mat1_rows, mat1_cols);
		REQUIRE(mat1.rows() == mat1_rows);
		REQUIRE(mat1.cols() == mat1_cols);
		REQUIRE(mat1.size() == mat1_size);
		
		mat1.identity();

		REQUIRE(approxEqual(mat1, Cumat::Matrixf::identity(mat1_rows, mat1_cols)));
	}

	SECTION("Rectangular identity matrix fill (rows > cols)")
	{
		size_t mat1_rows = 2048;
		size_t mat1_cols = 485;
		size_t mat1_size = mat1_rows * mat1_cols;

		Cumat::Matrixf mat1(mat1_rows, mat1_cols);
		REQUIRE(mat1.rows() == mat1_rows);
		REQUIRE(mat1.cols() == mat1_cols);
		REQUIRE(mat1.size() == mat1_size);
		
		mat1.identity();

		REQUIRE(approxEqual(mat1, Cumat::Matrixf::identity(mat1_rows, mat1_cols)));
	}

	SECTION("Rectangular identity matrix fill (rows < cols)")
	{
		size_t mat1_rows = 321;
		size_t mat1_cols = 4096;
		size_t mat1_size = mat1_rows * mat1_cols;

		Cumat::Matrixf mat1(mat1_rows, mat1_cols);
		REQUIRE(mat1.rows() == mat1_rows);
		REQUIRE(mat1.cols() == mat1_cols);
		REQUIRE(mat1.size() == mat1_size);
		
		mat1.identity();

		REQUIRE(approxEqual(mat1, Cumat::Matrixf::identity(mat1_rows, mat1_cols)));
	}

	Cumat::end();
}

TEST_CASE("Double identity matrix", "[identity][double]")
{
	Cumat::init();

	SECTION("Identity matrix assignment")
	{
		size_t mat1_rows = 1045;
		size_t mat1_cols = 500;
		size_t mat1_size = mat1_rows * mat1_cols;

		Cumat::Matrixd mat1 = Cumat::Matrixd::identity(mat1_rows, mat1_cols);
		REQUIRE(mat1.rows() == mat1_rows);
		REQUIRE(mat1.cols() == mat1_cols);
		REQUIRE(mat1.size() == mat1_size);

		for (size_t i = 0; i < mat1.rows(); ++i) {
			for (size_t j = 0; j < mat1.cols(); ++j) {
				if (i == j)
					REQUIRE(mat1(i, j) == Approx(1.0));
				else
					REQUIRE(mat1(i, j) == Approx(0.0));
			}
		}
	}

	SECTION("Square identity matrix fill")
	{
		size_t mat1_rows = 1024;
		size_t mat1_cols = 1024;
		size_t mat1_size = mat1_rows * mat1_cols;

		Cumat::Matrixd mat1(mat1_rows, mat1_cols);
		REQUIRE(mat1.rows() == mat1_rows);
		REQUIRE(mat1.cols() == mat1_cols);
		REQUIRE(mat1.size() == mat1_size);
		
		mat1.identity();

		REQUIRE(approxEqual(mat1, Cumat::Matrixd::identity(mat1_rows, mat1_cols)));
	}

	SECTION("Rectangular identity matrix fill (rows > cols)")
	{
		size_t mat1_rows = 2048;
		size_t mat1_cols = 485;
		size_t mat1_size = mat1_rows * mat1_cols;

		Cumat::Matrixd mat1(mat1_rows, mat1_cols);
		REQUIRE(mat1.rows() == mat1_rows);
		REQUIRE(mat1.cols() == mat1_cols);
		REQUIRE(mat1.size() == mat1_size);
		
		mat1.identity();

		REQUIRE(approxEqual(mat1, Cumat::Matrixd::identity(mat1_rows, mat1_cols)));
	}

	SECTION("Rectangular identity matrix fill (rows < cols)")
	{
		size_t mat1_rows = 321;
		size_t mat1_cols = 4096;
		size_t mat1_size = mat1_rows * mat1_cols;

		Cumat::Matrixd mat1(mat1_rows, mat1_cols);
		REQUIRE(mat1.rows() == mat1_rows);
		REQUIRE(mat1.cols() == mat1_cols);
		REQUIRE(mat1.size() == mat1_size);
		
		mat1.identity();

		REQUIRE(approxEqual(mat1, Cumat::Matrixd::identity(mat1_rows, mat1_cols)));
	}

	Cumat::end();
}
