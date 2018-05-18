#include "catch.hpp"
#include "util/GPUCompare.hpp"
#include "libcumat.h"
#include <cstdlib>
#include <time.h>

TEST_CASE("Float matrix copy", "[copy][float]")
{
	Cumat::init();

	size_t mat1_rows = ((double)std::rand() / (double)RAND_MAX) * 1000 + 1;
	size_t mat1_cols = ((double)std::rand() / (double)RAND_MAX) * 1000 + 1;
	size_t mat1_size = mat1_rows * mat1_cols;

	Cumat::Matrixf mat1 = Cumat::Matrixf::random(mat1_rows, mat1_cols);
	REQUIRE(mat1.rows() == mat1_rows);
	REQUIRE(mat1.cols() == mat1_cols);
	REQUIRE(mat1.size() == mat1_size);

	SECTION("Copy device vector (same size) (same type)")
	{
		thrust::device_vector<float> vec(mat1_size);

		for (size_t i = 0; i < vec.size(); ++i)
			vec[i] = ((double)std::rand() / (double)RAND_MAX) * 2 - 1;

		Cumat::Matrixf mat2(vec);
		mat2.resize(mat1_rows, mat1_cols);

		mat1.copy(vec.begin(), vec.end());

		REQUIRE(mat1.rows() == mat1_rows);
		REQUIRE(mat1.cols() == mat1_cols);
		REQUIRE(mat1.size() == mat1_size);
		REQUIRE(approxEqual(mat1, mat2));
	}

	SECTION("Copy device vector (same size) (different type)")
	{
		thrust::device_vector<double> vec(mat1_size);

		for (size_t i = 0; i < vec.size(); ++i)
			vec[i] = ((double)std::rand() / (double)RAND_MAX) * 2 - 1;

		mat1.copy(vec.begin(), vec.end());

		Cumat::Matrixf mat2(vec);
		mat2.resize(mat1_rows, mat1_cols);

		REQUIRE(mat1.rows() == mat1_rows);
		REQUIRE(mat1.cols() == mat1_cols);
		REQUIRE(mat1.size() == mat1_size);
		REQUIRE(approxEqual(mat1, mat2));
	}

	SECTION("Copy device vector (smaller size)")
	{
		size_t new_size = mat1_size * 0.65;
		thrust::device_vector<float> vec(new_size);
		
		Cumat::Matrixf mat2 = mat1;
		REQUIRE(approxEqual(mat1, mat2));

		for (size_t i = 0; i < vec.size(); ++i)
			vec[i] = ((double)std::rand() / (double)RAND_MAX) * 2 - 1;

		mat1.copy(vec.begin(), vec.end());

		REQUIRE(mat1.rows() == mat1_rows);
		REQUIRE(mat1.cols() == mat1_cols);
		REQUIRE(mat1.size() == mat1_size);

		for (size_t i = 0; i < mat1.size(); ++i) {
			if (i < vec.size())
				CHECK(mat1(i) == Approx(vec[i]));
			else
				CHECK(mat1(i) == Approx(mat2(i)));
		}
	}

	SECTION("Copy device vector (larger size)")
	{
		size_t new_size = mat1_size * 3.4;
		thrust::device_vector<float> vec(new_size);

		for (size_t i = 0; i < vec.size(); ++i)
			vec[i] = ((double)std::rand() / (double)RAND_MAX) * 2 - 1;

		mat1.copy(vec.begin(), vec.end());

		REQUIRE(mat1.rows() == 1);
		REQUIRE(mat1.cols() == new_size);
		REQUIRE(mat1.size() == new_size);
		REQUIRE(approxEqual(mat1, Cumat::Matrixf(vec)));
	}

	SECTION("Copy C++ vector (same size) (same type)")
	{
		std::vector<float> vec(mat1_size);

		for (size_t i = 0; i < vec.size(); ++i)
			vec[i] = ((double)std::rand() / (double)RAND_MAX) * 2 - 1;

		Cumat::Matrixf mat2(vec);
		mat2.resize(mat1_rows, mat1_cols);

		mat1.copy(vec.begin(), vec.end());

		REQUIRE(mat1.rows() == mat1_rows);
		REQUIRE(mat1.cols() == mat1_cols);
		REQUIRE(mat1.size() == mat1_size);
		REQUIRE(approxEqual(mat1, mat2));
	}

	SECTION("Copy C++ vector (same size) (different type)")
	{
		std::vector<double> vec(mat1_size);

		for (size_t i = 0; i < vec.size(); ++i)
			vec[i] = ((double)std::rand() / (double)RAND_MAX) * 2 - 1;

		mat1.copy(vec.begin(), vec.end());

		Cumat::Matrixf mat2(vec);
		mat2.resize(mat1_rows, mat1_cols);

		REQUIRE(mat1.rows() == mat1_rows);
		REQUIRE(mat1.cols() == mat1_cols);
		REQUIRE(mat1.size() == mat1_size);
		REQUIRE(approxEqual(mat1, mat2));
	}

	SECTION("Copy C++ vector (smaller size)")
	{
		size_t new_size = mat1_size * 0.65;
		std::vector<float> vec(new_size);
		
		Cumat::Matrixf mat2 = mat1;
		REQUIRE(approxEqual(mat1, mat2));

		for (size_t i = 0; i < vec.size(); ++i)
			vec[i] = ((double)std::rand() / (double)RAND_MAX) * 2 - 1;

		mat1.copy(vec.begin(), vec.end());

		REQUIRE(mat1.rows() == mat1_rows);
		REQUIRE(mat1.cols() == mat1_cols);
		REQUIRE(mat1.size() == mat1_size);

		for (size_t i = 0; i < mat1.size(); ++i) {
			if (i < vec.size())
				CHECK(mat1(i) == Approx(vec[i]));
			else
				CHECK(mat1(i) == Approx(mat2(i)));
		}
	}

	SECTION("Copy C++ vector (larger size)")
	{
		size_t new_size = mat1_size * 3.4;
		std::vector<float> vec(new_size);

		for (size_t i = 0; i < vec.size(); ++i)
			vec[i] = ((double)std::rand() / (double)RAND_MAX) * 2 - 1;

		mat1.copy(vec.begin(), vec.end());

		REQUIRE(mat1.rows() == 1);
		REQUIRE(mat1.cols() == new_size);
		REQUIRE(mat1.size() == new_size);
		REQUIRE(approxEqual(mat1, Cumat::Matrixf(vec)));
	}

	Cumat::end();
}

TEST_CASE("Double matrix copy", "[copy][double]")
{
	Cumat::init();

	size_t mat1_rows = ((double)std::rand() / (double)RAND_MAX) * 1000 + 1;
	size_t mat1_cols = ((double)std::rand() / (double)RAND_MAX) * 1000 + 1;
	size_t mat1_size = mat1_rows * mat1_cols;

	Cumat::Matrixd mat1 = Cumat::Matrixd::random(mat1_rows, mat1_cols);
	REQUIRE(mat1.rows() == mat1_rows);
	REQUIRE(mat1.cols() == mat1_cols);
	REQUIRE(mat1.size() == mat1_size);

	SECTION("Copy device vector (same size) (same type)")
	{
		thrust::device_vector<double> vec(mat1_size);

		for (size_t i = 0; i < vec.size(); ++i)
			vec[i] = ((double)std::rand() / (double)RAND_MAX) * 2 - 1;

		Cumat::Matrixd mat2(vec);
		mat2.resize(mat1_rows, mat1_cols);

		mat1.copy(vec.begin(), vec.end());

		REQUIRE(mat1.rows() == mat1_rows);
		REQUIRE(mat1.cols() == mat1_cols);
		REQUIRE(mat1.size() == mat1_size);
		REQUIRE(approxEqual(mat1, mat2));
	}

	SECTION("Copy device vector (same size) (different type)")
	{
		thrust::device_vector<float> vec(mat1_size);

		for (size_t i = 0; i < vec.size(); ++i)
			vec[i] = ((double)std::rand() / (double)RAND_MAX) * 2 - 1;

		mat1.copy(vec.begin(), vec.end());

		Cumat::Matrixd mat2(vec);
		mat2.resize(mat1_rows, mat1_cols);

		REQUIRE(mat1.rows() == mat1_rows);
		REQUIRE(mat1.cols() == mat1_cols);
		REQUIRE(mat1.size() == mat1_size);
		REQUIRE(approxEqual(mat1, mat2));
	}

	SECTION("Copy device vector (smaller size)")
	{
		size_t new_size = mat1_size * 0.65;
		thrust::device_vector<double> vec(new_size);
		
		Cumat::Matrixd mat2 = mat1;
		REQUIRE(approxEqual(mat1, mat2));

		for (size_t i = 0; i < vec.size(); ++i)
			vec[i] = ((double)std::rand() / (double)RAND_MAX) * 2 - 1;

		mat1.copy(vec.begin(), vec.end());

		REQUIRE(mat1.rows() == mat1_rows);
		REQUIRE(mat1.cols() == mat1_cols);
		REQUIRE(mat1.size() == mat1_size);

		for (size_t i = 0; i < mat1.size(); ++i) {
			if (i < vec.size())
				CHECK(mat1(i) == Approx(vec[i]));
			else
				CHECK(mat1(i) == Approx(mat2(i)));
		}
	}

	SECTION("Copy device vector (larger size)")
	{
		size_t new_size = mat1_size * 3.4;
		thrust::device_vector<double> vec(new_size);

		for (size_t i = 0; i < vec.size(); ++i)
			vec[i] = ((double)std::rand() / (double)RAND_MAX) * 2 - 1;

		mat1.copy(vec.begin(), vec.end());

		REQUIRE(mat1.rows() == 1);
		REQUIRE(mat1.cols() == new_size);
		REQUIRE(mat1.size() == new_size);
		REQUIRE(approxEqual(mat1, Cumat::Matrixd(vec)));
	}

	SECTION("Copy C++ vector (same size) (same type)")
	{
		std::vector<double> vec(mat1_size);

		for (size_t i = 0; i < vec.size(); ++i)
			vec[i] = ((double)std::rand() / (double)RAND_MAX) * 2 - 1;

		Cumat::Matrixd mat2(vec);
		mat2.resize(mat1_rows, mat1_cols);

		mat1.copy(vec.begin(), vec.end());

		REQUIRE(mat1.rows() == mat1_rows);
		REQUIRE(mat1.cols() == mat1_cols);
		REQUIRE(mat1.size() == mat1_size);
		REQUIRE(approxEqual(mat1, mat2));
	}

	SECTION("Copy C++ vector (same size) (different type)")
	{
		std::vector<double> vec(mat1_size);

		for (size_t i = 0; i < vec.size(); ++i)
			vec[i] = ((double)std::rand() / (double)RAND_MAX) * 2 - 1;

		mat1.copy(vec.begin(), vec.end());

		Cumat::Matrixd mat2(vec);
		mat2.resize(mat1_rows, mat1_cols);

		REQUIRE(mat1.rows() == mat1_rows);
		REQUIRE(mat1.cols() == mat1_cols);
		REQUIRE(mat1.size() == mat1_size);
		REQUIRE(approxEqual(mat1, mat2));
	}

	SECTION("Copy C++ vector (smaller size)")
	{
		size_t new_size = mat1_size * 0.65;
		std::vector<double> vec(new_size);
		
		Cumat::Matrixd mat2 = mat1;
		REQUIRE(approxEqual(mat1, mat2));

		for (size_t i = 0; i < vec.size(); ++i)
			vec[i] = ((double)std::rand() / (double)RAND_MAX) * 2 - 1;

		mat1.copy(vec.begin(), vec.end());

		REQUIRE(mat1.rows() == mat1_rows);
		REQUIRE(mat1.cols() == mat1_cols);
		REQUIRE(mat1.size() == mat1_size);

		for (size_t i = 0; i < mat1.size(); ++i) {
			if (i < vec.size())
				CHECK(mat1(i) == Approx(vec[i]));
			else
				CHECK(mat1(i) == Approx(mat2(i)));
		}
	}

	SECTION("Copy C++ vector (larger size)")
	{
		size_t new_size = mat1_size * 3.4;
		std::vector<double> vec(new_size);

		for (size_t i = 0; i < vec.size(); ++i)
			vec[i] = ((double)std::rand() / (double)RAND_MAX) * 2 - 1;

		mat1.copy(vec.begin(), vec.end());

		REQUIRE(mat1.rows() == 1);
		REQUIRE(mat1.cols() == new_size);
		REQUIRE(mat1.size() == new_size);
		REQUIRE(approxEqual(mat1, Cumat::Matrixd(vec)));
	}

	Cumat::end();
}
