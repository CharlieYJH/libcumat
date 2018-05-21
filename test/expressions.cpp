#include "catch.hpp"
#include "util/GPUCompare.hpp"
#include "libcumat.h"
#include <cstdlib>
#include <time.h>

TEST_CASE("Matrix expressions", "[operators][expressions]")
{
	size_t mat1_rows = ((double)std::rand() / (double)RAND_MAX) * 500 + 1;
	size_t mat1_cols = ((double)std::rand() / (double)RAND_MAX) * 500 + 1;
	size_t mat1_size = mat1_rows * mat1_cols;

	Cumat::Matrixf mat1 = Cumat::Matrixf::random(mat1_rows, mat1_cols);
	REQUIRE(mat1.rows() == mat1_rows);
	REQUIRE(mat1.cols() == mat1_cols);
	REQUIRE(mat1.size() == mat1_size);
	REQUIRE_FALSE(approxEqual(mat1, 0));

	Cumat::Matrixf mat2 = Cumat::Matrixf::random(mat1_rows, mat1_cols);
	REQUIRE(mat2.rows() == mat1_rows);
	REQUIRE(mat2.cols() == mat1_cols);
	REQUIRE(mat2.size() == mat1_size);
	REQUIRE_FALSE(approxEqual(mat2, 0));

	Cumat::Matrixd mat3 = Cumat::Matrixf::random(mat1_rows, mat1_cols);
	REQUIRE(mat3.rows() == mat1_rows);
	REQUIRE(mat3.cols() == mat1_cols);
	REQUIRE(mat3.size() == mat1_size);
	REQUIRE_FALSE(approxEqual(mat3, 0));

	Cumat::Matrixd mat4 = Cumat::Matrixf::random(mat1_rows, mat1_cols);
	REQUIRE(mat4.rows() == mat1_rows);
	REQUIRE(mat4.cols() == mat1_cols);
	REQUIRE(mat4.size() == mat1_size);
	REQUIRE_FALSE(approxEqual(mat4, 0));

	SECTION("Expression 1")
	{
		Cumat::Matrixf mat5(mat1_rows, mat1_cols);
		REQUIRE(mat5.rows() == mat1_rows);
		REQUIRE(mat5.cols() == mat1_cols);
		REQUIRE(mat5.size() == mat1_size);
		REQUIRE(approxEqual(mat5, 0));

		Cumat::Matrixf mat6;
		REQUIRE(mat6.rows() == 0);
		REQUIRE(mat6.cols() == 0);
		REQUIRE(mat6.size() == 0);

		mat6 = 2.3 + mat1 + mat2 * 8.5 - -mat4;

		for (size_t i = 0; i < mat5.size(); ++i)
			mat5(i) = 2.3 + mat1(i).val() + mat2(i).val() * 8.5 - -mat4(i).val();

		REQUIRE(approxEqual(mat5, mat6));
	}

	SECTION("Expression 2")
	{
		Cumat::Matrixf mat5(mat1_rows, mat1_cols);
		REQUIRE(mat5.rows() == mat1_rows);
		REQUIRE(mat5.cols() == mat1_cols);
		REQUIRE(mat5.size() == mat1_size);
		REQUIRE(approxEqual(mat5, 0));

		Cumat::Matrixf mat6;
		REQUIRE(mat6.rows() == 0);
		REQUIRE(mat6.cols() == 0);
		REQUIRE(mat6.size() == 0);

		mat6 = tanh(mat1 * 0.8f) + transpose(exp(transpose(mat3) + transpose(mat2)));

		for (size_t i = 0; i < mat5.size(); ++i)
			mat5(i) = std::tanh(mat1(i).val() * 0.8f) + std::exp(mat3(i).val() + mat2(i).val());

		REQUIRE(approxEqual(mat5, mat6));
	}

	SECTION("Expression 3")
	{
		Cumat::Matrixf mat5(mat1_rows, mat1_cols);
		REQUIRE(mat5.rows() == mat1_rows);
		REQUIRE(mat5.cols() == mat1_cols);
		REQUIRE(mat5.size() == mat1_size);
		REQUIRE(approxEqual(mat5, 0));

		mat1.transpose();
		REQUIRE(mat1.rows() == mat1_cols);
		REQUIRE(mat1.cols() == mat1_rows);
		REQUIRE(mat1.size() == mat1_size);

		Cumat::Matrixf mat6;
		REQUIRE(mat6.rows() == 0);
		REQUIRE(mat6.cols() == 0);
		REQUIRE(mat6.size() == 0);

		mat6 = (transpose(mat1) + 4.5) * pow(mat3 + 2, mat2) / 2.4f;

		for (size_t i = 0; i < mat5.rows(); ++i) {
			for (size_t j = 0; j < mat5.cols(); ++j) {
				mat5(i, j) = (mat1(j, i).val() + 4.5) * std::pow(mat3(i, j).val() + 2, mat2(i, j).val()) / 2.4f;
			}
		}

		REQUIRE(approxEqual(mat5, mat6));
	}

	SECTION("Expression 4")
	{
		Cumat::Matrixf mat5(mat1_rows, mat1_cols);
		REQUIRE(mat5.rows() == mat1_rows);
		REQUIRE(mat5.cols() == mat1_cols);
		REQUIRE(mat5.size() == mat1_size);
		REQUIRE(approxEqual(mat5, 0));

		Cumat::Matrixf mat6;
		REQUIRE(mat6.rows() == 0);
		REQUIRE(mat6.cols() == 0);
		REQUIRE(mat6.size() == 0);

		mat6 = atan2(2 * mat1, 3.5 / (mat2 + 3)) + 2.3 + powf(2.0, mat1);

		for (size_t i = 0; i < mat5.size(); ++i)
			mat5(i) = std::atan2(2 * mat1(i).val(), 3.5 / (mat2(i).val() + 3)) + 2.3 + std::pow(2.0, mat1(i).val());

		REQUIRE(approxEqual(mat5, mat6));
	}

	SECTION("Expression 5")
	{
		Cumat::Matrixf mat5(mat1_rows, mat1_cols);
		REQUIRE(mat5.rows() == mat1_rows);
		REQUIRE(mat5.cols() == mat1_cols);
		REQUIRE(mat5.size() == mat1_size);
		REQUIRE(approxEqual(mat5, 0));

		Cumat::Matrixf mat6;
		REQUIRE(mat6.rows() == 0);
		REQUIRE(mat6.cols() == 0);
		REQUIRE(mat6.size() == 0);

		mat6 = cbrt(square(4.15f / sigmoid(mat2 * mat1)));

		for (size_t i = 0; i < mat5.size(); ++i) {
            float denom = 1.0f / (1.0f + std::exp(-mat2(i).val() * mat1(i).val()));
			mat5(i) = std::cbrt((4.15f / denom) * (4.15f / denom));
        }

		REQUIRE(approxEqual(mat5, mat6));
	}

	SECTION("Expression 6")
	{
		Cumat::Matrixf mat5(mat1_rows, mat1_cols);
		REQUIRE(mat5.rows() == mat1_rows);
		REQUIRE(mat5.cols() == mat1_cols);
		REQUIRE(mat5.size() == mat1_size);
		REQUIRE(approxEqual(mat5, 0));

		Cumat::Matrixf mat6;
		REQUIRE(mat6.rows() == 0);
		REQUIRE(mat6.cols() == 0);
		REQUIRE(mat6.size() == 0);

		mat6 = transpose(log(transpose(abs(mat1 - 4.5))) + exp2(1.2 * transpose(mat3) + transpose(mat4 / 3)));

		for (size_t i = 0; i < mat5.size(); ++i)
			mat5(i) = std::log(std::abs(mat1(i).val() - 4.5)) + std::exp2(1.2 * mat3(i).val() + mat4(i).val() / 3);

		REQUIRE(approxEqual(mat5, mat6));
	}

	SECTION("Expression 7")
	{
		Cumat::Matrixf mat5(mat1_rows, mat1_cols);
		REQUIRE(mat5.rows() == mat1_rows);
		REQUIRE(mat5.cols() == mat1_cols);
		REQUIRE(mat5.size() == mat1_size);
		REQUIRE(approxEqual(mat5, 0));

		Cumat::Matrixf mat6;
		REQUIRE(mat6.rows() == 0);
		REQUIRE(mat6.cols() == 0);
		REQUIRE(mat6.size() == 0);

		mat6 = atan2f(4, mat1 / mat1) + powf(mat3, 2) + max(mat1 + 2, mat2 + 2) - minf(2.0, mat2);

		for (size_t i = 0; i < mat5.size(); ++i)
			mat5(i) = std::atan2(4, 1) + std::pow(mat3(i).val(), 2) + std::max(mat1(i).val() + 2, mat2(i).val() + 2) - std::min(2.0f, mat2(i).val());

		REQUIRE(approxEqual(mat5, mat6));
	}

	SECTION("Expression 8")
	{
		Cumat::Matrixf mat5(mat1_rows, mat1_cols);
		REQUIRE(mat5.rows() == mat1_rows);
		REQUIRE(mat5.cols() == mat1_cols);
		REQUIRE(mat5.size() == mat1_size);
		REQUIRE(approxEqual(mat5, 0));

		Cumat::Matrixf mat6;
		REQUIRE(mat6.rows() == 0);
		REQUIRE(mat6.cols() == 0);
		REQUIRE(mat6.size() == 0);

		mat6 = maxf(4.0f, mat1 + mat2 + min(mat3, mat2)) + max(abs(mat1 * mat2), 0.3) - powf(abs(mat2), mat3);

		for (size_t i = 0; i < mat5.size(); ++i)
			mat5(i) = std::max(4.0f, mat1(i).val() + mat2(i).val() + std::min((float)mat3(i), mat2(i).val())) + std::max(std::abs(mat1(i).val() * mat2(i).val()), 0.3f) - std::pow(std::abs(mat2(i).val()), mat3(i).val());

		REQUIRE(approxEqual(mat5, mat6));
	}

	SECTION("Expression 9")
	{
		Cumat::Matrixf mat5(mat1_rows, mat1_cols);
		REQUIRE(mat5.rows() == mat1_rows);
		REQUIRE(mat5.cols() == mat1_cols);
		REQUIRE(mat5.size() == mat1_size);
		REQUIRE(approxEqual(mat5, 0));

		Cumat::Matrixf mat6;
		REQUIRE(mat6.rows() == 0);
		REQUIRE(mat6.cols() == 0);
		REQUIRE(mat6.size() == 0);

		mat6 = pow(0.3, mat1) + pow(mat1, 2) + maxf(mat1 + mat1, abs(mat2 - mat3)) - maxf(mat4, 0.32) - 3.4 + 0.3f;

		for (size_t i = 0; i < mat5.size(); ++i)
			mat5(i) = std::pow(0.3, mat1(i).val()) + std::pow(mat1(i).val(), 2) + std::max((float)mat1(i).val() * 2, (float)std::abs(mat2(i).val() - mat3(i).val())) - std::max((float)mat4(i).val(), 0.32f) - 3.4 + 0.3f;

		REQUIRE(approxEqual(mat5, mat6));
	}

	SECTION("Expression 10")
	{
		Cumat::Matrixf mat5(mat1_rows, mat1_cols);
		REQUIRE(mat5.rows() == mat1_rows);
		REQUIRE(mat5.cols() == mat1_cols);
		REQUIRE(mat5.size() == mat1_size);
		REQUIRE(approxEqual(mat5, 0));

		Cumat::Matrixf mat6;
		REQUIRE(mat6.rows() == 0);
		REQUIRE(mat6.cols() == 0);
		REQUIRE(mat6.size() == 0);

		mat6 = max(0.3, rcbrt(mat1)) + minf(minf(mat2, mat3), -0.4f);

		for (size_t i = 0; i < mat5.size(); ++i)
			mat5(i) = std::max(0.3, 1.0 / std::cbrt(mat1(i).val())) + std::min(std::min((float)mat2(i), (float)mat3(i)), -0.4f);

		REQUIRE(approxEqual(mat5, mat6));
	}

	SECTION("Expression 11")
	{
		Cumat::Matrixf mat5(mat1_rows, mat1_cols);
		REQUIRE(mat5.rows() == mat1_rows);
		REQUIRE(mat5.cols() == mat1_cols);
		REQUIRE(mat5.size() == mat1_size);
		REQUIRE(approxEqual(mat5, 0));

		Cumat::Matrixf mat6;
		REQUIRE(mat6.rows() == 0);
		REQUIRE(mat6.cols() == 0);
		REQUIRE(mat6.size() == 0);

		mat6 = min(sin(mat3), 0.4) + min(-0.34f, sinh(mat1 + mat2));

		for (size_t i = 0; i < mat5.size(); ++i)
			mat5(i) = std::min(std::sin(mat3(i).val()), 0.4) + std::min(-0.34f, std::sinh(mat1(i).val() + mat2(i).val()));

		REQUIRE(approxEqual(mat5, mat6));
	}

	SECTION("Expression 12")
	{
		Cumat::Matrixf mat5(mat1_rows, mat1_cols);
		REQUIRE(mat5.rows() == mat1_rows);
		REQUIRE(mat5.cols() == mat1_cols);
		REQUIRE(mat5.size() == mat1_size);
		REQUIRE(approxEqual(mat5, 0));

		Cumat::Matrixf mat6;
		REQUIRE(mat6.rows() == 0);
		REQUIRE(mat6.cols() == 0);
		REQUIRE(mat6.size() == 0);

		mat6 = atan2(2.4f, mat3 * 4) + atan2(-log2(abs(mat1)), -0.31);

		for (size_t i = 0; i < mat5.size(); ++i)
			mat5(i) = std::atan2(2.4f, mat3(i).val() * 4) + std::atan2(-std::log2(std::abs(mat1(i).val())), -0.31);

		REQUIRE(approxEqual(mat5, mat6));
	}

	SECTION("Expression 13")
	{
		Cumat::Matrixf mat5(mat1_rows, mat1_cols);
		REQUIRE(mat5.rows() == mat1_rows);
		REQUIRE(mat5.cols() == mat1_cols);
		REQUIRE(mat5.size() == mat1_size);
		REQUIRE(approxEqual(mat5, 0));

		Cumat::Matrixf mat6;
		REQUIRE(mat6.rows() == 0);
		REQUIRE(mat6.cols() == 0);
		REQUIRE(mat6.size() == 0);

		mat6 = atan2f(mat3 * mat2 / 3, mat1 - mat4) + atan2f(ceil(mat1 + mat2), 5.4f);

		for (size_t i = 0; i < mat5.size(); ++i)
			mat5(i) = std::atan2(mat3(i).val() * mat2(i).val() / 3, mat1(i).val() - mat4(i).val()) + std::atan2(std::ceil(mat1(i).val() + mat2(i).val()), 5.4f);

		REQUIRE(approxEqual(mat5, mat6));
	}
}
