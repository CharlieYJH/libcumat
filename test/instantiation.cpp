#include "catch.hpp"
#include "util/GPUCompare.hpp"
#include "libcumat.h"
#include <cstdlib>
#include <time.h>

TEST_CASE("Float matrix empty instantiation", "[empty][instantiation][float]")
{
	Cumat::Matrixf mat1;
	REQUIRE(mat1.rows() == 0);
	REQUIRE(mat1.cols() == 0);
	REQUIRE(mat1.size() == 0);
}

TEST_CASE("Double matrix empty instantiation", "[empty][instantiation][double]")
{
	Cumat::Matrixd mat1;
	REQUIRE(mat1.rows() == 0);
	REQUIRE(mat1.cols() == 0);
	REQUIRE(mat1.size() == 0);
}

TEST_CASE("Float matrix direct value instantiation", "[direct][instantiation][float]")
{
	Cumat::Matrixf mat1(450, 120);
	REQUIRE(mat1.rows() == 450);
	REQUIRE(mat1.cols() == 120);
	REQUIRE(mat1.size() == 450 * 120);
	REQUIRE(approxEqual(mat1, 0));
}

TEST_CASE("Double matrix direct value instantiation", "[direct][instantiation][double]")
{
	Cumat::Matrixd mat1(1020, 293);
	REQUIRE(mat1.rows() == 1020);
	REQUIRE(mat1.cols() == 293);
	REQUIRE(mat1.size() == 1020 * 293);
	REQUIRE(approxEqual(mat1, 0));
}

TEST_CASE("Float matrix direct value instantiation (with floating point)", "[direct][instantiation][float]")
{
	Cumat::Matrixf mat1(1030.38281, 1921.192);
	REQUIRE(mat1.rows() == 1030);
	REQUIRE(mat1.cols() == 1921);
	REQUIRE(mat1.size() == 1030 * 1921);
	REQUIRE(approxEqual(mat1, 0));
}

TEST_CASE("Double matrix direct value instantiation (with floating point)", "[direct][instantiation][double]")
{
	Cumat::Matrixd mat1(2918.372, 45.128);
	REQUIRE(mat1.rows() == 2918);
	REQUIRE(mat1.cols() == 45);
	REQUIRE(mat1.size() == 2918 * 45);
	REQUIRE(approxEqual(mat1, 0));
}

TEST_CASE("Float matrix size instantiation", "[size][instantiation][float]")
{
	SECTION("Rows > 0 and Cols > 0")
	{
		srand(time(0));

		size_t mat1_rows = ((double)std::rand() / (double)RAND_MAX) * 3000 + 1;
		size_t mat1_cols = ((double)std::rand() / (double)RAND_MAX) * 3000 + 1;
		size_t mat1_size = mat1_rows * mat1_cols;

		REQUIRE(mat1_size > 0);

		Cumat::Matrixf mat1(mat1_rows, mat1_cols);
		REQUIRE(mat1.rows() == mat1_rows);
		REQUIRE(mat1.cols() == mat1_cols);
		REQUIRE(mat1.size() == mat1_size);
	}

	SECTION("Rows = 0 and Cols > 0")
	{
		size_t mat1_rows = 0;
		size_t mat1_cols = ((double)std::rand() / (double)RAND_MAX) * 3000 + 1;

		Cumat::Matrixf mat1(mat1_rows, mat1_cols);
		REQUIRE(mat1.rows() == 0);
		REQUIRE(mat1.cols() == 0);
		REQUIRE(mat1.size() == 0);
	}

	SECTION("Rows > 0 and Cols = 0")
	{
		size_t mat1_rows = ((double)std::rand() / (double)RAND_MAX) * 3000 + 1;
		size_t mat1_cols = 0;

		Cumat::Matrixf mat1(mat1_rows, mat1_cols);
		REQUIRE(mat1.rows() == 0);
		REQUIRE(mat1.cols() == 0);
		REQUIRE(mat1.size() == 0);
	}
}

TEST_CASE("Double matrix size instantiation", "[size][instantiation][double]")
{
	SECTION("Rows > 0 and Cols > 0")
	{
		size_t mat1_rows = ((double)std::rand() / (double)RAND_MAX) * 3000 + 1;
		size_t mat1_cols = ((double)std::rand() / (double)RAND_MAX) * 3000 + 1;
		size_t mat1_size = mat1_rows * mat1_cols;

		REQUIRE(mat1_size > 0);

		Cumat::Matrixd mat1(mat1_rows, mat1_cols);
		REQUIRE(mat1.rows() == mat1_rows);
		REQUIRE(mat1.cols() == mat1_cols);
		REQUIRE(mat1.size() == mat1_size);
	}

	SECTION("Rows = 0 and Cols > 0")
	{
		size_t mat1_rows = 0;
		size_t mat1_cols = ((double)std::rand() / (double)RAND_MAX) * 3000 + 1;

		Cumat::Matrixd mat1(mat1_rows, mat1_cols);
		REQUIRE(mat1.rows() == 0);
		REQUIRE(mat1.cols() == 0);
		REQUIRE(mat1.size() == 0);
	}

	SECTION("Rows > 0 and Cols = 0")
	{
		size_t mat1_rows = ((double)std::rand() / (double)RAND_MAX) * 3000 + 1;
		size_t mat1_cols = 0;

		Cumat::Matrixd mat1(mat1_rows, mat1_cols);
		REQUIRE(mat1.rows() == 0);
		REQUIRE(mat1.cols() == 0);
		REQUIRE(mat1.size() == 0);
	}
}

TEST_CASE("Float matrix assignment instantiation", "[assignment][instantiation][float]")
{
	size_t mat1_rows = ((double)std::rand() / (double)RAND_MAX) * 3000 + 1;
	size_t mat1_cols = ((double)std::rand() / (double)RAND_MAX) * 3000 + 1;
	size_t mat1_size = mat1_rows * mat1_cols;

	REQUIRE(mat1_size > 0);

	Cumat::Matrixf mat1(mat1_rows, mat1_cols);
	REQUIRE(mat1.rows() == mat1_rows);
	REQUIRE(mat1.cols() == mat1_cols);
	REQUIRE(mat1.size() == mat1_size);

	mat1.rand();
	REQUIRE_FALSE(approxEqual(mat1, 0));

	Cumat::Matrixf mat2 = mat1;
	REQUIRE(mat2.rows() == mat1.rows());
	REQUIRE(mat2.cols() == mat1.cols());
	REQUIRE(mat2.size() == mat1.size());
	REQUIRE(approxEqual(mat2, mat1));
}

TEST_CASE("Double matrix assignment instantiation", "[assignment][instantiation][double]")
{
	size_t mat1_rows = ((double)std::rand() / (double)RAND_MAX) * 3000 + 1;
	size_t mat1_cols = ((double)std::rand() / (double)RAND_MAX) * 3000 + 1;
	size_t mat1_size = mat1_rows * mat1_cols;

	REQUIRE(mat1_size > 0);

	Cumat::Matrixd mat1(mat1_rows, mat1_cols);
	REQUIRE(mat1.rows() == mat1_rows);
	REQUIRE(mat1.cols() == mat1_cols);
	REQUIRE(mat1.size() == mat1_size);

	mat1.rand();
	REQUIRE_FALSE(approxEqual(mat1, 0));

	Cumat::Matrixd mat2 = mat1;
	REQUIRE(mat2.rows() == mat1.rows());
	REQUIRE(mat2.cols() == mat1.cols());
	REQUIRE(mat2.size() == mat1.size());
	REQUIRE(approxEqual(mat2, mat1));
}

TEST_CASE("Float matrix assignment instantiation (different type)", "[assignment][instantiation][float]")
{
	size_t mat1_rows = ((double)std::rand() / (double)RAND_MAX) * 3000 + 1;
	size_t mat1_cols = ((double)std::rand() / (double)RAND_MAX) * 3000 + 1;
	size_t mat1_size = mat1_rows * mat1_cols;

	REQUIRE(mat1_size > 0);

	Cumat::Matrixd mat1(mat1_rows, mat1_cols);
	REQUIRE(mat1.rows() == mat1_rows);
	REQUIRE(mat1.cols() == mat1_cols);
	REQUIRE(mat1.size() == mat1_size);

	mat1.rand();
	REQUIRE_FALSE(approxEqual(mat1, 0));

	Cumat::Matrixf mat2 = mat1;
	REQUIRE(mat2.rows() == mat1.rows());
	REQUIRE(mat2.cols() == mat1.cols());
	REQUIRE(mat2.size() == mat1.size());
	REQUIRE(approxEqual(mat2, mat1));
}

TEST_CASE("Double matrix assignment instantiation (different type)", "[assignment][instantiation][double]")
{
	size_t mat1_rows = ((double)std::rand() / (double)RAND_MAX) * 3000 + 1;
	size_t mat1_cols = ((double)std::rand() / (double)RAND_MAX) * 3000 + 1;
	size_t mat1_size = mat1_rows * mat1_cols;

	REQUIRE(mat1_size > 0);

	Cumat::Matrixf mat1(mat1_rows, mat1_cols);
	REQUIRE(mat1.rows() == mat1_rows);
	REQUIRE(mat1.cols() == mat1_cols);
	REQUIRE(mat1.size() == mat1_size);

	mat1.rand();
	REQUIRE_FALSE(approxEqual(mat1, 0));

	Cumat::Matrixd mat2 = mat1;
	REQUIRE(mat2.rows() == mat1.rows());
	REQUIRE(mat2.cols() == mat1.cols());
	REQUIRE(mat2.size() == mat1.size());
	REQUIRE(approxEqual(mat2, mat1));
}

TEST_CASE("Float matrix thrust device vector instantiation", "[thrust][device_vector][instantiation][float]")
{
	size_t mat1_size = ((double)std::rand() / (double)RAND_MAX) * 3000 + 1;

	SECTION("Zero size assignment")
	{
		thrust::device_vector<float> empty_vec1;
		thrust::device_vector<double> empty_vec2;

		Cumat::Matrixf mat1(empty_vec1);
		REQUIRE(mat1.rows() == 0);
		REQUIRE(mat1.cols() == 0);
		REQUIRE(mat1.size() == 0);
		REQUIRE(mat1.thrustVector().size() == 0);

		Cumat::Matrixf mat2(empty_vec2);
		REQUIRE(mat2.rows() == 0);
		REQUIRE(mat2.cols() == 0);
		REQUIRE(mat2.size() == 0);
		REQUIRE(mat2.thrustVector().size() == 0);

		Cumat::Matrixf mat3(empty_vec2.begin(), empty_vec2.end());
		REQUIRE(mat3.rows() == 0);
		REQUIRE(mat3.cols() == 0);
		REQUIRE(mat3.size() == 0);
		REQUIRE(mat3.thrustVector().size() == 0);
	}

	SECTION("Float to float assignment")
	{
		thrust::device_vector<float> vec1(mat1_size);
		REQUIRE(vec1.size() > 0);

		for (size_t i = 0; i < vec1.size(); ++i)
			vec1[i] = (float)std::rand() / (float)RAND_MAX;

		Cumat::Matrixf mat1(vec1);
		REQUIRE(mat1.rows() == 1);
		REQUIRE(mat1.cols() == vec1.size());
		REQUIRE(mat1.size() == vec1.size());

		for (size_t i = 0; i < mat1.size(); ++i)
			CHECK(mat1(i) == Approx(vec1[i]));
	}

	SECTION("Double to float assignment")
	{
		thrust::device_vector<double> vec1(mat1_size);
		REQUIRE(vec1.size() > 0);

		for (size_t i = 0; i < vec1.size(); ++i)
			vec1[i] = (double)std::rand() / (double)RAND_MAX;

		Cumat::Matrixf mat1(vec1);
		REQUIRE(mat1.rows() == 1);
		REQUIRE(mat1.cols() == vec1.size());
		REQUIRE(mat1.size() == vec1.size());

		for (size_t i = 0; i < mat1.size(); ++i)
			CHECK(mat1(i) == Approx(vec1[i]));
	}

	SECTION("Iterator assignment")
	{
		thrust::device_vector<float> vec1(mat1_size);
		REQUIRE(vec1.size() > 0);

		for (size_t i = 0; i < vec1.size(); ++i)
			vec1[i] = (double)std::rand() / (double)RAND_MAX;

		Cumat::Matrixf mat1(vec1.begin(), vec1.end());
		REQUIRE(mat1.rows() == 1);
		REQUIRE(mat1.cols() == vec1.size());
		REQUIRE(mat1.size() == vec1.size());

		for (size_t i = 0; i < mat1.size(); ++i)
			CHECK(mat1(i) == Approx(vec1[i]));
	}
}

TEST_CASE("Double matrix thrust device vector instantiation", "[thrust][device_vector][instantiation][double]")
{
	size_t mat1_size = ((double)std::rand() / (double)RAND_MAX) * 3000 + 1;

	SECTION("Zero size assignment")
	{
		thrust::device_vector<float> empty_vec1;
		thrust::device_vector<double> empty_vec2;

		Cumat::Matrixd mat1(empty_vec1);
		REQUIRE(mat1.rows() == 0);
		REQUIRE(mat1.cols() == 0);
		REQUIRE(mat1.size() == 0);
		REQUIRE(mat1.thrustVector().size() == 0);

		Cumat::Matrixd mat2(empty_vec2);
		REQUIRE(mat2.rows() == 0);
		REQUIRE(mat2.cols() == 0);
		REQUIRE(mat2.size() == 0);
		REQUIRE(mat2.thrustVector().size() == 0);

		Cumat::Matrixd mat3(empty_vec2.begin(), empty_vec2.end());
		REQUIRE(mat3.rows() == 0);
		REQUIRE(mat3.cols() == 0);
		REQUIRE(mat3.size() == 0);
		REQUIRE(mat3.thrustVector().size() == 0);
	}

	SECTION("Float to double assignment")
	{
		thrust::device_vector<float> vec1(mat1_size);
		REQUIRE(vec1.size() > 0);

		for (size_t i = 0; i < vec1.size(); ++i)
			vec1[i] = (float)std::rand() / (float)RAND_MAX;

		Cumat::Matrixd mat1(vec1);
		REQUIRE(mat1.rows() == 1);
		REQUIRE(mat1.cols() == vec1.size());
		REQUIRE(mat1.size() == vec1.size());

		for (size_t i = 0; i < mat1.size(); ++i)
			CHECK(mat1(i) == Approx(vec1[i]));
	}

	SECTION("Double to double assignment")
	{
		thrust::device_vector<double> vec1(mat1_size);
		REQUIRE(vec1.size() > 0);

		for (size_t i = 0; i < vec1.size(); ++i)
			vec1[i] = (double)std::rand() / (double)RAND_MAX;

		Cumat::Matrixd mat1(vec1);
		REQUIRE(mat1.rows() == 1);
		REQUIRE(mat1.cols() == vec1.size());
		REQUIRE(mat1.size() == vec1.size());

		for (size_t i = 0; i < mat1.size(); ++i)
			CHECK(mat1(i) == Approx(vec1[i]));
	}

	SECTION("Iterator assignment")
	{
		thrust::device_vector<float> vec1(mat1_size);
		REQUIRE(vec1.size() > 0);

		for (size_t i = 0; i < vec1.size(); ++i)
			vec1[i] = (double)std::rand() / (double)RAND_MAX;

		Cumat::Matrixf mat1(vec1.begin(), vec1.end());
		REQUIRE(mat1.rows() == 1);
		REQUIRE(mat1.cols() == vec1.size());
		REQUIRE(mat1.size() == vec1.size());

		for (size_t i = 0; i < mat1.size(); ++i)
			CHECK(mat1(i) == Approx(vec1[i]));
	}
}

TEST_CASE("Float matrix thrust host vector instantiation", "[thrust][host_vector][instantiation][float]")
{
	size_t mat1_size = ((double)std::rand() / (double)RAND_MAX) * 3000 + 1;

	SECTION("Zero size assignment")
	{
		thrust::host_vector<float> empty_vec1;
		thrust::host_vector<double> empty_vec2;

		Cumat::Matrixf mat1(empty_vec1);
		REQUIRE(mat1.rows() == 0);
		REQUIRE(mat1.cols() == 0);
		REQUIRE(mat1.size() == 0);
		REQUIRE(mat1.thrustVector().size() == 0);

		Cumat::Matrixf mat2(empty_vec2);
		REQUIRE(mat2.rows() == 0);
		REQUIRE(mat2.cols() == 0);
		REQUIRE(mat2.size() == 0);
		REQUIRE(mat2.thrustVector().size() == 0);

		Cumat::Matrixf mat3(empty_vec2.begin(), empty_vec2.end());
		REQUIRE(mat3.rows() == 0);
		REQUIRE(mat3.cols() == 0);
		REQUIRE(mat3.size() == 0);
		REQUIRE(mat3.thrustVector().size() == 0);
	}

	SECTION("Float to float assignment")
	{
		thrust::host_vector<float> vec1(mat1_size);
		REQUIRE(vec1.size() > 0);

		for (size_t i = 0; i < vec1.size(); ++i)
			vec1[i] = (float)std::rand() / (float)RAND_MAX;

		Cumat::Matrixf mat1(vec1);
		REQUIRE(mat1.rows() == 1);
		REQUIRE(mat1.cols() == vec1.size());
		REQUIRE(mat1.size() == vec1.size());

		for (size_t i = 0; i < mat1.size(); ++i)
			CHECK(mat1(i) == Approx(vec1[i]));
	}

	SECTION("Double to float assignment")
	{
		thrust::host_vector<double> vec1(mat1_size);
		REQUIRE(vec1.size() > 0);

		for (size_t i = 0; i < vec1.size(); ++i)
			vec1[i] = (double)std::rand() / (double)RAND_MAX;

		Cumat::Matrixf mat1(vec1);
		REQUIRE(mat1.rows() == 1);
		REQUIRE(mat1.cols() == vec1.size());
		REQUIRE(mat1.size() == vec1.size());

		for (size_t i = 0; i < mat1.size(); ++i)
			CHECK(mat1(i) == Approx(vec1[i]));
	}

	SECTION("Iterator assignment")
	{
		thrust::host_vector<float> vec1(mat1_size);
		REQUIRE(vec1.size() > 0);

		for (size_t i = 0; i < vec1.size(); ++i)
			vec1[i] = (double)std::rand() / (double)RAND_MAX;

		Cumat::Matrixf mat1(vec1.begin(), vec1.end());
		REQUIRE(mat1.rows() == 1);
		REQUIRE(mat1.cols() == vec1.size());
		REQUIRE(mat1.size() == vec1.size());

		for (size_t i = 0; i < mat1.size(); ++i)
			CHECK(mat1(i) == Approx(vec1[i]));
	}
}

TEST_CASE("Double matrix thrust host vector instantiation", "[thrust][host_vector][instantiation][double]")
{
	size_t mat1_size = ((double)std::rand() / (double)RAND_MAX) * 3000 + 1;

	SECTION("Zero size assignment")
	{
		thrust::host_vector<float> empty_vec1;
		thrust::host_vector<double> empty_vec2;

		Cumat::Matrixd mat1(empty_vec1);
		REQUIRE(mat1.rows() == 0);
		REQUIRE(mat1.cols() == 0);
		REQUIRE(mat1.size() == 0);
		REQUIRE(mat1.thrustVector().size() == 0);

		Cumat::Matrixd mat2(empty_vec2);
		REQUIRE(mat2.rows() == 0);
		REQUIRE(mat2.cols() == 0);
		REQUIRE(mat2.size() == 0);
		REQUIRE(mat2.thrustVector().size() == 0);

		Cumat::Matrixd mat3(empty_vec2.begin(), empty_vec2.end());
		REQUIRE(mat3.rows() == 0);
		REQUIRE(mat3.cols() == 0);
		REQUIRE(mat3.size() == 0);
		REQUIRE(mat3.thrustVector().size() == 0);
	}

	SECTION("Float to double assignment")
	{
		thrust::host_vector<float> vec1(mat1_size);
		REQUIRE(vec1.size() > 0);

		for (size_t i = 0; i < vec1.size(); ++i)
			vec1[i] = (float)std::rand() / (float)RAND_MAX;

		Cumat::Matrixd mat1(vec1);
		REQUIRE(mat1.rows() == 1);
		REQUIRE(mat1.cols() == vec1.size());
		REQUIRE(mat1.size() == vec1.size());

		for (size_t i = 0; i < mat1.size(); ++i)
			CHECK(mat1(i) == Approx(vec1[i]));
	}

	SECTION("Double to double assignment")
	{
		thrust::host_vector<double> vec1(mat1_size);
		REQUIRE(vec1.size() > 0);

		for (size_t i = 0; i < vec1.size(); ++i)
			vec1[i] = (double)std::rand() / (double)RAND_MAX;

		Cumat::Matrixd mat1(vec1);
		REQUIRE(mat1.rows() == 1);
		REQUIRE(mat1.cols() == vec1.size());
		REQUIRE(mat1.size() == vec1.size());

		for (size_t i = 0; i < mat1.size(); ++i)
			CHECK(mat1(i) == Approx(vec1[i]));
	}

	SECTION("Iterator assignment")
	{
		thrust::host_vector<float> vec1(mat1_size);
		REQUIRE(vec1.size() > 0);

		for (size_t i = 0; i < vec1.size(); ++i)
			vec1[i] = (double)std::rand() / (double)RAND_MAX;

		Cumat::Matrixf mat1(vec1.begin(), vec1.end());
		REQUIRE(mat1.rows() == 1);
		REQUIRE(mat1.cols() == vec1.size());
		REQUIRE(mat1.size() == vec1.size());

		for (size_t i = 0; i < mat1.size(); ++i)
			CHECK(mat1(i) == Approx(vec1[i]));
	}
}

TEST_CASE("Float matrix C++ vector instantiation", "[vector][instantiation][float]")
{
	size_t mat1_size = ((double)std::rand() / (double)RAND_MAX) * 3000 + 1;

	SECTION("Zero size assignment")
	{
		std::vector<float> empty_vec1;
		std::vector<double> empty_vec2;

		Cumat::Matrixf mat1(empty_vec1);
		REQUIRE(mat1.rows() == 0);
		REQUIRE(mat1.cols() == 0);
		REQUIRE(mat1.size() == 0);
		REQUIRE(mat1.thrustVector().size() == 0);

		Cumat::Matrixf mat2(empty_vec2);
		REQUIRE(mat2.rows() == 0);
		REQUIRE(mat2.cols() == 0);
		REQUIRE(mat2.size() == 0);
		REQUIRE(mat2.thrustVector().size() == 0);

		Cumat::Matrixf mat3(empty_vec2.begin(), empty_vec2.end());
		REQUIRE(mat3.rows() == 0);
		REQUIRE(mat3.cols() == 0);
		REQUIRE(mat3.size() == 0);
		REQUIRE(mat3.thrustVector().size() == 0);
	}

	SECTION("Float to float assignment")
	{
		std::vector<float> vec1(mat1_size);
		REQUIRE(vec1.size() > 0);

		for (size_t i = 0; i < vec1.size(); ++i)
			vec1[i] = (float)std::rand() / (float)RAND_MAX;

		Cumat::Matrixf mat1(vec1);
		REQUIRE(mat1.rows() == 1);
		REQUIRE(mat1.cols() == vec1.size());
		REQUIRE(mat1.size() == vec1.size());

		for (size_t i = 0; i < mat1.size(); ++i)
			CHECK(mat1(i) == Approx(vec1[i]));
	}

	SECTION("Double to float assignment")
	{
		std::vector<double> vec1(mat1_size);
		REQUIRE(vec1.size() > 0);

		for (size_t i = 0; i < vec1.size(); ++i)
			vec1[i] = (double)std::rand() / (double)RAND_MAX;

		Cumat::Matrixf mat1(vec1);
		REQUIRE(mat1.rows() == 1);
		REQUIRE(mat1.cols() == vec1.size());
		REQUIRE(mat1.size() == vec1.size());

		for (size_t i = 0; i < mat1.size(); ++i)
			CHECK(mat1(i) == Approx(vec1[i]));
	}

	SECTION("Iterator assignment")
	{
		std::vector<float> vec1(mat1_size);
		REQUIRE(vec1.size() > 0);

		for (size_t i = 0; i < vec1.size(); ++i)
			vec1[i] = (double)std::rand() / (double)RAND_MAX;

		Cumat::Matrixf mat1(vec1.begin(), vec1.end());
		REQUIRE(mat1.rows() == 1);
		REQUIRE(mat1.cols() == vec1.size());
		REQUIRE(mat1.size() == vec1.size());

		for (size_t i = 0; i < mat1.size(); ++i)
			CHECK(mat1(i) == Approx(vec1[i]));
	}
}

TEST_CASE("Double matrix C++ vector instantiation", "[vector][instantiation][double]")
{
	size_t mat1_size = ((double)std::rand() / (double)RAND_MAX) * 3000 + 1;

	SECTION("Zero size assignment")
	{
		std::vector<float> empty_vec1;
		std::vector<double> empty_vec2;

		Cumat::Matrixd mat1(empty_vec1);
		REQUIRE(mat1.rows() == 0);
		REQUIRE(mat1.cols() == 0);
		REQUIRE(mat1.size() == 0);
		REQUIRE(mat1.thrustVector().size() == 0);

		Cumat::Matrixd mat2(empty_vec2);
		REQUIRE(mat2.rows() == 0);
		REQUIRE(mat2.cols() == 0);
		REQUIRE(mat2.size() == 0);
		REQUIRE(mat2.thrustVector().size() == 0);

		Cumat::Matrixd mat3(empty_vec2.begin(), empty_vec2.end());
		REQUIRE(mat3.rows() == 0);
		REQUIRE(mat3.cols() == 0);
		REQUIRE(mat3.size() == 0);
		REQUIRE(mat3.thrustVector().size() == 0);
	}

	SECTION("Float to double assignment")
	{
		std::vector<float> vec1(mat1_size);
		REQUIRE(vec1.size() > 0);

		for (size_t i = 0; i < vec1.size(); ++i)
			vec1[i] = (float)std::rand() / (float)RAND_MAX;

		Cumat::Matrixd mat1(vec1);
		REQUIRE(mat1.rows() == 1);
		REQUIRE(mat1.cols() == vec1.size());
		REQUIRE(mat1.size() == vec1.size());

		for (size_t i = 0; i < mat1.size(); ++i)
			CHECK(mat1(i) == Approx(vec1[i]));
	}

	SECTION("Double to double assignment")
	{
		std::vector<double> vec1(mat1_size);
		REQUIRE(vec1.size() > 0);

		for (size_t i = 0; i < vec1.size(); ++i)
			vec1[i] = (double)std::rand() / (double)RAND_MAX;

		Cumat::Matrixd mat1(vec1);
		REQUIRE(mat1.rows() == 1);
		REQUIRE(mat1.cols() == vec1.size());
		REQUIRE(mat1.size() == vec1.size());

		for (size_t i = 0; i < mat1.size(); ++i)
			CHECK(mat1(i) == Approx(vec1[i]));
	}

	SECTION("Iterator assignment")
	{
		std::vector<float> vec1(mat1_size);
		REQUIRE(vec1.size() > 0);

		for (size_t i = 0; i < vec1.size(); ++i)
			vec1[i] = (double)std::rand() / (double)RAND_MAX;

		Cumat::Matrixd mat1(vec1.begin(), vec1.end());
		REQUIRE(mat1.rows() == 1);
		REQUIRE(mat1.cols() == vec1.size());
		REQUIRE(mat1.size() == vec1.size());

		for (size_t i = 0; i < mat1.size(); ++i)
			CHECK(mat1(i) == Approx(vec1[i]));
	}
}

TEST_CASE("Float matrix fill instantiation", "[fill][instantiation][float]")
{
	size_t mat1_rows = ((double)std::rand() / (double)RAND_MAX) * 3000 + 1;
	size_t mat1_cols = ((double)std::rand() / (double)RAND_MAX) * 3000 + 1;
	size_t mat1_size = mat1_rows * mat1_cols;
	float mat1_val = 2.30;

	Cumat::Matrixf mat1(mat1_rows, mat1_cols, mat1_val);
	REQUIRE(mat1.rows() == mat1_rows);
	REQUIRE(mat1.cols() == mat1_cols);
	REQUIRE(mat1.size() == mat1_size);
	REQUIRE(approxEqual(mat1, mat1_val));
}

TEST_CASE("Double matrix fill instantiation", "[fill][instantiation][double]")
{
	size_t mat1_rows = ((double)std::rand() / (double)RAND_MAX) * 3000 + 1;
	size_t mat1_cols = ((double)std::rand() / (double)RAND_MAX) * 3000 + 1;
	size_t mat1_size = mat1_rows * mat1_cols;
	double mat1_val = 4.56102;

	Cumat::Matrixd mat1(mat1_rows, mat1_cols, mat1_val);
	REQUIRE(mat1.rows() == mat1_rows);
	REQUIRE(mat1.cols() == mat1_cols);
	REQUIRE(mat1.size() == mat1_size);
	REQUIRE(approxEqual(mat1, mat1_val));
}

TEST_CASE("Float matrix expression instantiation", "[expression][instantiation][float]")
{
	size_t mat1_rows = ((double)std::rand() / (double)RAND_MAX) * 3000 + 1;
	size_t mat1_cols = ((double)std::rand() / (double)RAND_MAX) * 3000 + 1;
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
}

TEST_CASE("Double matrix expression instantiation", "[expression][instantiation][double]")
{
	size_t mat1_rows = ((double)std::rand() / (double)RAND_MAX) * 3000 + 1;
	size_t mat1_cols = ((double)std::rand() / (double)RAND_MAX) * 3000 + 1;
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
}
