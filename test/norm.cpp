#include "catch.hpp"
#include "GPUCompare.hpp"
#include "Core"
#include <cstdlib>
#include <time.h>

TEST_CASE("Float matrix norm", "[norm][float]")
{
	Cumat::init();

	srand(time(0));

	size_t mat1_rows = ((double)std::rand() / (double)RAND_MAX) * 2000;
	size_t mat1_cols = ((double)std::rand() / (double)RAND_MAX) * 2000;
	size_t mat1_size = mat1_rows * mat1_cols;

	Cumat::Matrixf mat1 = Cumat::Matrixf::random(mat1_rows, mat1_cols);

	REQUIRE(mat1.rows() == mat1_rows);
	REQUIRE(mat1.cols() == mat1_cols);
	REQUIRE(mat1.size() == mat1_size);
	REQUIRE_FALSE(approxEqual(mat1, 0.0f));

	float norm = 0.0f;

	for (size_t i = 0; i < mat1.size(); ++i) {
		float val = mat1(i);
		norm += val * val;
	}

	norm = std::sqrt(norm);

	REQUIRE(mat1.norm() == Approx(norm).epsilon(0.05));

	Cumat::end();
}

TEST_CASE("Double matrix norm", "[norm][double]")
{
	Cumat::init();

	srand(time(0));

	size_t mat1_rows = ((double)std::rand() / (double)RAND_MAX) * 2000;
	size_t mat1_cols = ((double)std::rand() / (double)RAND_MAX) * 2000;
	size_t mat1_size = mat1_rows * mat1_cols;

	Cumat::Matrixd mat1 = Cumat::Matrixd::random(mat1_rows, mat1_cols);

	REQUIRE(mat1.rows() == mat1_rows);
	REQUIRE(mat1.cols() == mat1_cols);
	REQUIRE(mat1.size() == mat1_size);
	REQUIRE_FALSE(approxEqual(mat1, 0.0));

	double norm = 0.0f;

	for (size_t i = 0; i < mat1.size(); ++i) {
		double val = mat1(i);
		norm += val * val;
	}

	norm = std::sqrt(norm);

	REQUIRE(mat1.norm() == Approx(norm));

	Cumat::end();
}
