#include "catch.hpp"
#include "util/CPUMatrix.hpp"
#include "Core"

TEST_CASE("Float matrix basic instantiation", "[basic][instantiation][float]")
{
	Cumat::Matrixf mat1;
	REQUIRE(mat1.rows() == 0);
	REQUIRE(mat1.cols() == 0);
	REQUIRE(mat1.size() == 0);

	size_t mat2_rows = 125;
	size_t mat2_cols = 230;
	size_t mat2_size = mat2_rows * mat2_cols;

	Cumat::Matrixf mat2(mat2_rows, mat2_cols);
	REQUIRE(mat2.rows() == mat2_rows);
	REQUIRE(mat2.cols() == mat2_cols);
	REQUIRE(mat2.size() == mat2_size);

	size_t mat3_rows = 1203;
	size_t mat3_cols = 2301;
	size_t mat3_size = mat3_rows * mat3_cols;

	Cumat::Matrixf mat3(mat3_rows, mat3_cols);
	REQUIRE(mat3.rows() == mat3_rows);
	REQUIRE(mat3.cols() == mat3_cols);
	REQUIRE(mat3.size() == mat3_size);

	Cumat::Matrixf mat4 = mat3;
	REQUIRE(mat4.rows() == mat3.rows());
	REQUIRE(mat4.cols() == mat3.cols());
	REQUIRE(mat4.size() == mat3.size());
}

TEST_CASE("Double matrix basic instantiation", "[basic][instantiation][double]")
{
	Cumat::Matrixd mat1;
	REQUIRE(mat1.rows() == 0);
	REQUIRE(mat1.cols() == 0);
	REQUIRE(mat1.size() == 0);

	size_t mat2_rows = 250;
	size_t mat2_cols = 650;
	size_t mat2_size = mat2_rows * mat2_cols;

	Cumat::Matrixd mat2(mat2_rows, mat2_cols);
	REQUIRE(mat2.rows() == mat2_rows);
	REQUIRE(mat2.cols() == mat2_cols);
	REQUIRE(mat2.size() == mat2_size);

	size_t mat3_rows = 1250;
	size_t mat3_cols = 1020;
	size_t mat3_size = mat3_rows * mat3_cols;

	Cumat::Matrixd mat3(mat3_rows, mat3_cols);
	REQUIRE(mat3.rows() == mat3_rows);
	REQUIRE(mat3.cols() == mat3_cols);
	REQUIRE(mat3.size() == mat3_size);

	Cumat::Matrixf mat4 = mat3;
	REQUIRE(mat4.rows() == mat3.rows());
	REQUIRE(mat4.cols() == mat3.cols());
	REQUIRE(mat4.size() == mat3.size());
}

TEST_CASE("Float matrix fill instantiation.", "[fill][instantiation][float]")
{
	size_t mat1_rows = 250;
	size_t mat1_cols = 665;
	size_t mat1_size = mat1_rows * mat1_cols;
	float mat1_val = 2.30;

	Cumat::Matrixf mat1(mat1_rows, mat1_cols, mat1_val);
	REQUIRE(mat1.rows() == mat1_rows);
	REQUIRE(mat1.cols() == mat1_cols);
	REQUIRE(mat1.size() == mat1_size);

	for (size_t i = 0; i < mat1.size(); ++i)
		REQUIRE(mat1(i) == Approx(mat1_val));
}

TEST_CASE("double matrix fill instantiation.", "[fill][instantiation][double]")
{
	size_t mat1_rows = 128;
	size_t mat1_cols = 240;
	size_t mat1_size = mat1_rows * mat1_cols;
	double mat1_val = 4.56102;

	Cumat::Matrixd mat1(mat1_rows, mat1_cols, mat1_val);
	REQUIRE(mat1.rows() == mat1_rows);
	REQUIRE(mat1.cols() == mat1_cols);
	REQUIRE(mat1.size() == mat1_size);

	for (size_t i = 0; i < mat1.size(); ++i)
		REQUIRE(mat1(i) == Approx(mat1_val));
}
