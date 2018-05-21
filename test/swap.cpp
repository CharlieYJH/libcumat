#include "catch.hpp"
#include "util/GPUCompare.hpp"
#include "libcumat.h"

TEST_CASE("Float matrix swap", "[swap][float]")
{
	size_t mat1_rows = 250;
	size_t mat1_cols = 120;
	size_t mat1_size = mat1_rows * mat1_cols;

	Cumat::Matrixf mat1 = Cumat::Matrixf::random(mat1_rows, mat1_cols);
	REQUIRE(mat1.rows() == mat1_rows);
	REQUIRE(mat1.cols() == mat1_cols);
	REQUIRE(mat1.size() == mat1_size);

	size_t mat2_rows = 4500;
	size_t mat2_cols = 1020;
	size_t mat2_size = mat2_rows * mat2_cols;

	Cumat::Matrixf mat2 = Cumat::Matrixf::random(mat2_rows, mat2_cols);
	REQUIRE(mat2.rows() == mat2_rows);
	REQUIRE(mat2.cols() == mat2_cols);
	REQUIRE(mat2.size() == mat2_size);

	Cumat::Matrixf mat3 = mat1;
	REQUIRE(approxEqual(mat3, mat1));
	
	Cumat::Matrixf mat4 = mat2;
	REQUIRE(approxEqual(mat4, mat2));

	mat3.swap(mat4);

	REQUIRE(approxEqual(mat3, mat2));
	REQUIRE_FALSE(approxEqual(mat3, mat1));

	REQUIRE(approxEqual(mat4, mat1));
	REQUIRE_FALSE(approxEqual(mat4, mat2));

	mat4.swap(mat3);

	REQUIRE(approxEqual(mat3, mat1));
	REQUIRE_FALSE(approxEqual(mat3, mat2));

	REQUIRE(approxEqual(mat4, mat2));
	REQUIRE_FALSE(approxEqual(mat4, mat1));
}

TEST_CASE("Double matrix swap", "[swap][float]")
{
	size_t mat1_rows = 250;
	size_t mat1_cols = 120;
	size_t mat1_size = mat1_rows * mat1_cols;

	Cumat::Matrixd mat1 = Cumat::Matrixd::random(mat1_rows, mat1_cols);
	REQUIRE(mat1.rows() == mat1_rows);
	REQUIRE(mat1.cols() == mat1_cols);
	REQUIRE(mat1.size() == mat1_size);

	size_t mat2_rows = 4500;
	size_t mat2_cols = 1020;
	size_t mat2_size = mat2_rows * mat2_cols;

	Cumat::Matrixd mat2 = Cumat::Matrixd::random(mat2_rows, mat2_cols);
	REQUIRE(mat2.rows() == mat2_rows);
	REQUIRE(mat2.cols() == mat2_cols);
	REQUIRE(mat2.size() == mat2_size);

	Cumat::Matrixd mat3 = mat1;
	REQUIRE(approxEqual(mat3, mat1));
	
	Cumat::Matrixd mat4 = mat2;
	REQUIRE(approxEqual(mat4, mat2));

	mat3.swap(mat4);

	REQUIRE(approxEqual(mat3, mat2));
	REQUIRE_FALSE(approxEqual(mat3, mat1));

	REQUIRE(approxEqual(mat4, mat1));
	REQUIRE_FALSE(approxEqual(mat4, mat2));

	mat4.swap(mat3);

	REQUIRE(approxEqual(mat3, mat1));
	REQUIRE_FALSE(approxEqual(mat3, mat2));

	REQUIRE(approxEqual(mat4, mat2));
	REQUIRE_FALSE(approxEqual(mat4, mat1));
}
