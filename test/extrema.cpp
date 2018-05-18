#include "catch.hpp"
#include "util/GPUCompare.hpp"
#include "libcumat.h"
#include <cstdlib>
#include <time.h>
#include <limits>

TEST_CASE("Float matrix extrema", "[extrema][float]")
{
	Cumat::init();

	srand(time(0));

	size_t mat1_rows = ((double)std::rand() / (double)RAND_MAX) * 2000 + 1;
	size_t mat1_cols = ((double)std::rand() / (double)RAND_MAX) * 2000 + 1;
	size_t mat1_size = mat1_rows * mat1_cols;

	Cumat::Matrixf mat1 = Cumat::Matrixf::random(mat1_rows, mat1_cols);
	REQUIRE(mat1.rows() == mat1_rows);
	REQUIRE(mat1.cols() == mat1_cols);
	REQUIRE(mat1.size() == mat1_size);
	REQUIRE_FALSE(approxEqual(mat1, 0.0));

	float max_element = std::numeric_limits<float>::min();
	float min_element = std::numeric_limits<float>::max();
	size_t max_index = 0;
	size_t min_index = 0;

	for (size_t i = 0; i < mat1.size(); ++i) {

		float val = mat1(i);

		if (val > max_element) {
			max_element = val;
			max_index = i;
		}

		if (val < min_element) {
			min_element = val;
			min_index = i;
		}
	}

	REQUIRE(mat1.maxElement() == Approx(max_element));
	REQUIRE(mat1.maxIndex() == max_index);
	REQUIRE(mat1.minElement() == Approx(min_element));
	REQUIRE(mat1.minIndex() == min_index);

	Cumat::end();
}

TEST_CASE("Double matrix extrema", "[extrema][double]")
{
	Cumat::init();

	size_t mat1_rows = ((double)std::rand() / (double)RAND_MAX) * 2000 + 1;
	size_t mat1_cols = ((double)std::rand() / (double)RAND_MAX) * 2000 + 1;
	size_t mat1_size = mat1_rows * mat1_cols;

	Cumat::Matrixd mat1 = Cumat::Matrixd::random(mat1_rows, mat1_cols);
	REQUIRE(mat1.rows() == mat1_rows);
	REQUIRE(mat1.cols() == mat1_cols);
	REQUIRE(mat1.size() == mat1_size);
	REQUIRE_FALSE(approxEqual(mat1, 0.0));

	double max_element = std::numeric_limits<double>::min();
	double min_element = std::numeric_limits<double>::max();
	size_t max_index = 0;
	size_t min_index = 0;

	for (size_t i = 0; i < mat1.size(); ++i) {

		double val = mat1(i);

		if (val > max_element) {
			max_element = val;
			max_index = i;
		}

		if (val < min_element) {
			min_element = val;
			min_index = i;
		}
	}

	REQUIRE(mat1.maxElement() == Approx(max_element));
	REQUIRE(mat1.maxIndex() == max_index);
	REQUIRE(mat1.minElement() == Approx(min_element));
	REQUIRE(mat1.minIndex() == min_index);

	Cumat::end();
}
