#include "catch.hpp"
#include "util/GPUCompare.hpp"
#include "Core"

TEST_CASE("Float matrix basic instantiation", "[basic][instantiation][float]")
{
	Cumat::init();

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

	Cumat::Matrixf mat4 = mat2;
	REQUIRE(mat4.rows() == mat2.rows());
	REQUIRE(mat4.cols() == mat2.cols());
	REQUIRE(mat4.size() == mat2.size());

	Cumat::end();
}

TEST_CASE("Double matrix basic instantiation", "[basic][instantiation][double]")
{
	Cumat::init();

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

	Cumat::Matrixd mat4 = mat2;
	REQUIRE(mat4.rows() == mat2.rows());
	REQUIRE(mat4.cols() == mat2.cols());
	REQUIRE(mat4.size() == mat2.size());
	
	Cumat::end();
}

TEST_CASE("Float matrix fill instantiation", "[fill][instantiation][float]")
{
	Cumat::init();

	size_t mat1_rows = 250;
	size_t mat1_cols = 665;
	size_t mat1_size = mat1_rows * mat1_cols;
	float mat1_val = 2.30;

	Cumat::Matrixf mat1(mat1_rows, mat1_cols, mat1_val);
	REQUIRE(mat1.rows() == mat1_rows);
	REQUIRE(mat1.cols() == mat1_cols);
	REQUIRE(mat1.size() == mat1_size);
	REQUIRE(approxEqual(mat1, mat1_val));

	Cumat::end();
}

TEST_CASE("double matrix fill instantiation", "[fill][instantiation][double]")
{
	Cumat::init();

	size_t mat1_rows = 128;
	size_t mat1_cols = 240;
	size_t mat1_size = mat1_rows * mat1_cols;
	double mat1_val = 4.56102;

	Cumat::Matrixd mat1(mat1_rows, mat1_cols, mat1_val);
	REQUIRE(mat1.rows() == mat1_rows);
	REQUIRE(mat1.cols() == mat1_cols);
	REQUIRE(mat1.size() == mat1_size);
	REQUIRE(approxEqual(mat1, mat1_val));

	Cumat::end();
}

TEST_CASE("Float matrix expression instantiation", "[expression][instantiation][float]")
{
	Cumat::init();

	size_t mat1_rows = 1020;
	size_t mat1_cols = 120;
	size_t mat1_size = mat1_rows * mat1_cols;

	Cumat::Matrixf mat1(mat1_rows, mat1_cols);
	REQUIRE(mat1.rows() == mat1_rows);
	REQUIRE(mat1.cols() == mat1_cols);
	REQUIRE(mat1.size() == mat1_size);

	Cumat::Matrixf mat2 = mat1;
	REQUIRE(mat2.rows() == mat1_rows);
	REQUIRE(mat2.cols() == mat1_cols);
	REQUIRE(mat2.size() == mat1_size);

	mat1.rand();
	mat2.rand();

	Cumat::Matrixf mat3 = 2 * mat1 + mat2 * mat2;
	REQUIRE(mat3.rows() == mat1_rows);
	REQUIRE(mat3.cols() == mat1_cols);
	REQUIRE(mat3.size() == mat1_size);
	REQUIRE(approxEqual(mat3, (2 * mat1 + mat2 * mat2).eval()));

	Cumat::end();
}

TEST_CASE("Double matrix expression instantiation", "[expression][instantiation][double]")
{
	Cumat::init();

	size_t mat1_rows = 250;
	size_t mat1_cols = 260;
	size_t mat1_size = mat1_rows * mat1_cols;

	Cumat::Matrixd mat1(mat1_rows, mat1_cols);
	REQUIRE(mat1.rows() == mat1_rows);
	REQUIRE(mat1.cols() == mat1_cols);
	REQUIRE(mat1.size() == mat1_size);

	Cumat::Matrixd mat2 = mat1;
	REQUIRE(mat2.rows() == mat1_rows);
	REQUIRE(mat2.cols() == mat1_cols);
	REQUIRE(mat2.size() == mat1_size);

	mat1.rand();
	mat2.rand();

	Cumat::Matrixd mat3 = 2.0f * mat1 + mat2;
	REQUIRE(mat3.rows() == mat1_rows);
	REQUIRE(mat3.cols() == mat1_cols);
	REQUIRE(mat3.size() == mat1_size);
	REQUIRE(approxEqual(mat3, (2.0f * mat1 + mat2).eval<double>()));

	Cumat::end();
}
