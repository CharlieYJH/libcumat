#include "catch.hpp"
#include "GPUCompare.hpp"
#include "Core"
#include <cstdlib>
#include <time.h>

TEST_CASE("Float matrix math ops", "[math][float]")
{
	Cumat::init();

	srand(time(0));

	size_t mat1_rows = ((double)std::rand() / (double)RAND_MAX) * 500 + 1;
	size_t mat1_cols = ((double)std::rand() / (double)RAND_MAX) * 500 + 1;
	size_t mat1_size = mat1_rows * mat1_cols;

	Cumat::Matrixf mat1 = Cumat::Matrixf::random(mat1_rows, mat1_cols);
	REQUIRE(mat1.rows() == mat1_rows);
	REQUIRE(mat1.cols() == mat1_cols);
	REQUIRE(mat1.size() == mat1_size);
	REQUIRE_FALSE(approxEqual(mat1, 0.0));

	Cumat::Matrixf mat2 = mat1;
	REQUIRE(approxEqual(mat1, mat2));

	SECTION("Absolute")
	{
		mat1.abs();

		for (size_t i = 0; i < mat2.size(); ++i) {

			float val = mat2(i);
			
			if (val < 0)
				mat2(i) = -val;
		}

		REQUIRE(approxEqual(mat1, mat2));
	}

	SECTION("Inverse")
	{
		mat1.inverse();

		for (size_t i = 0; i < mat2.size(); ++i)
			mat2(i) = 1.0f / mat2(i).val();

		REQUIRE(approxEqual(mat1, mat2));
	}

	SECTION("Clip")
	{
		float max = 0.2;
		float min = -0.5;

		mat1.clip(-0.5, 0.2);

		for (size_t i = 0; i < mat2.size(); ++i) {
			float val = mat2(i);
			mat2(i) = (val > max) ? max : (val < min) ? min : val;
		}

		REQUIRE(approxEqual(mat1, mat2));
	}

	SECTION("Exponentiation (base e)")
	{
		mat1.exp();

		for (size_t i = 0; i < mat2.size(); ++i) {
			float val = mat2(i);
			mat2(i) = std::exp(val);
		}

		REQUIRE(approxEqual(mat1, mat2));
	}

	SECTION("Exponentiation (base 10)")
	{
		mat1.exp10();

		for (size_t i = 0; i < mat2.size(); ++i) {
			float val = mat2(i);
			mat2(i) = std::pow(10, val);
		}

		REQUIRE(approxEqual(mat1, mat2));
	}

	SECTION("Exponentiation (base 2)")
	{
		mat1.exp2();

		for (size_t i = 0; i < mat2.size(); ++i) {
			float val = mat2(i);
			mat2(i) = std::exp2(val);
		}

		REQUIRE(approxEqual(mat1, mat2));
	}

	SECTION("Logarithm (base e)")
	{
		mat1.log();

		for (size_t i = 0; i < mat2.size(); ++i) {
			float val = mat2(i);
			mat2(i) = std::log(val);
		}

		REQUIRE(approxEqual(mat1, mat2));
	}

	SECTION("Logarithm (1 + x) (base e)")
	{
		mat1.log1p();

		for (size_t i = 0; i < mat2.size(); ++i) {
			float val = mat2(i);
			mat2(i) = std::log1p(val);
		}

		REQUIRE(approxEqual(mat1, mat2));
	}

	SECTION("Logarithm (base 10)")
	{
		mat1.log10();

		for (size_t i = 0; i < mat2.size(); ++i) {
			float val = mat2(i);
			mat2(i) = std::log10(val);
		}

		REQUIRE(approxEqual(mat1, mat2));
	}

	SECTION("Logarithm (base 2)")
	{
		mat1.log2();

		for (size_t i = 0; i < mat2.size(); ++i) {
			float val = mat2(i);
			mat2(i) = std::log2(val);
		}

		REQUIRE(approxEqual(mat1, mat2));
	}

	SECTION("Power (double)")
	{
		int power = ((double)std::rand() / (double)RAND_MAX) * 4;

		mat1.pow(power);

		for (size_t i = 0; i < mat2.size(); ++i) {
			float val = mat2(i);
			mat2(i) = std::pow(val, power);
		}

		REQUIRE(approxEqual(mat1, mat2));
	}

	SECTION("Power (float)")
	{
		int power = ((double)std::rand() / (double)RAND_MAX) * 4;

		mat1.powf(power);

		for (size_t i = 0; i < mat2.size(); ++i) {
			float val = mat2(i);
			mat2(i) = std::pow(val, power);
		}

		REQUIRE(approxEqual(mat1, mat2));
	}

	SECTION("Square")
	{
		mat1.square();

		for (size_t i = 0; i < mat2.size(); ++i) {
			float val = mat2(i);
			mat2(i) = val * val;
		}

		REQUIRE(approxEqual(mat1, mat2));
	}

	SECTION("Square root")
	{
		mat1.sqrt();

		for (size_t i = 0; i < mat2.size(); ++i) {
			float val = mat2(i);
			mat2(i) = std::sqrt(val);
		}

		REQUIRE(approxEqual(mat1, mat2));
	}

	SECTION("Inverse square root")
	{
		mat1.rsqrt();

		for (size_t i = 0; i < mat2.size(); ++i) {
			float val = mat2(i);
			mat2(i) = 1.0f / std::sqrt(val);
		}

		REQUIRE(approxEqual(mat1, mat2, 1e-4f));
	}

	SECTION("Cube")
	{
		mat1.cube();

		for (size_t i = 0; i < mat2.size(); ++i) {
			float val = mat2(i);
			mat2(i) = val * val * val;
		}

		REQUIRE(approxEqual(mat1, mat2));
	}

	SECTION("Cube root")
	{
		mat1.cbrt();

		for (size_t i = 0; i < mat2.size(); ++i) {
			float val = mat2(i);
			mat2(i) = std::cbrt(val);
		}

		REQUIRE(approxEqual(mat1, mat2));
	}

	SECTION("Inverse cube root")
	{
		mat1.rcbrt();

		for (size_t i = 0; i < mat2.size(); ++i) {
			float val = mat2(i);
			mat2(i) = 1.0f / std::cbrt(val);
		}

		REQUIRE(approxEqual(mat1, mat2));
	}

	SECTION("Sine")
	{
		mat1.sin();

		for (size_t i = 0; i < mat2.size(); ++i) {
			float val = mat2(i);
			mat2(i) = std::sin(val);
		}

		REQUIRE(approxEqual(mat1, mat2));
	}

	SECTION("Cosine")
	{
		mat1.cos();

		for (size_t i = 0; i < mat2.size(); ++i) {
			float val = mat2(i);
			mat2(i) = std::cos(val);
		}

		REQUIRE(approxEqual(mat1, mat2));
	}

	SECTION("Tangent")
	{
		mat1.tan();

		for (size_t i = 0; i < mat2.size(); ++i) {
			float val = mat2(i);
			mat2(i) = std::tan(val);
		}

		REQUIRE(approxEqual(mat1, mat2));
	}

	SECTION("Arcsine")
	{
		mat1.asin();

		for (size_t i = 0; i < mat2.size(); ++i) {
			float val = mat2(i);
			mat2(i) = std::asin(val);
		}

		REQUIRE(approxEqual(mat1, mat2));
	}

	SECTION("Arccosine")
	{
		mat1.acos();

		for (size_t i = 0; i < mat2.size(); ++i) {
			float val = mat2(i);
			mat2(i) = std::acos(val);
		}

		REQUIRE(approxEqual(mat1, mat2));
	}

	SECTION("Arctangent")
	{
		mat1.atan();

		for (size_t i = 0; i < mat2.size(); ++i) {
			float val = mat2(i);
			mat2(i) = std::atan(val);
		}

		REQUIRE(approxEqual(mat1, mat2));
	}

	SECTION("Hyperbolic sine")
	{
		mat1.sinh();

		for (size_t i = 0; i < mat2.size(); ++i) {
			float val = mat2(i);
			mat2(i) = std::sinh(val);
		}

		REQUIRE(approxEqual(mat1, mat2));
	}

	SECTION("Hyperbolic cosine")
	{
		mat1.cosh();

		for (size_t i = 0; i < mat2.size(); ++i) {
			float val = mat2(i);
			mat2(i) = std::cosh(val);
		}

		REQUIRE(approxEqual(mat1, mat2));
	}

	SECTION("Hyperbolic tangent")
	{
		mat1.tanh();

		for (size_t i = 0; i < mat2.size(); ++i) {
			float val = mat2(i);
			mat2(i) = std::tanh(val);
		}

		REQUIRE(approxEqual(mat1, mat2));
	}

	SECTION("Inverse hyperbolic sine")
	{
		mat1.asinh();

		for (size_t i = 0; i < mat2.size(); ++i) {
			float val = mat2(i);
			mat2(i) = std::asinh(val);
		}

		REQUIRE(approxEqual(mat1, mat2));
	}

	SECTION("Inverse hyperbolic cosine")
	{
		mat1.rand(1, 100);
		mat2 = mat1;
		REQUIRE(approxEqual(mat1, mat2));

		mat1.acosh();

		for (size_t i = 0; i < mat2.size(); ++i) {
			float val = mat2(i);
			mat2(i) = std::acosh(val);
		}

		REQUIRE(approxEqual(mat1, mat2));
	}

	SECTION("Inverse hyperbolic tangent")
	{
		mat1.atanh();

		for (size_t i = 0; i < mat2.size(); ++i) {
			float val = mat2(i);
			mat2(i) = std::atanh(val);
		}

		REQUIRE(approxEqual(mat1, mat2));
	}

	SECTION("Sigmoid")
	{
		mat1.sigmoid();

		for (size_t i = 0; i < mat2.size(); ++i) {
			float val = mat2(i);
			mat2(i) = 1.0f / (1 + std::exp(-val));
		}

		REQUIRE(approxEqual(mat1, mat2));
	}

	SECTION("Ceiling function")
	{
		mat1.ceil();

		for (size_t i = 0; i < mat2.size(); ++i) {
			float val = mat2(i);
			mat2(i) = std::ceil(val);
		}

		REQUIRE(approxEqual(mat1, mat2));
	}

	SECTION("Floor function")
	{
		mat1.floor();

		for (size_t i = 0; i < mat2.size(); ++i) {
			float val = mat2(i);
			mat2(i) = std::floor(val);
		}

		REQUIRE(approxEqual(mat1, mat2));
	}

	SECTION("Round")
	{
		mat1.rand(-1000, 1000);
		mat2 = mat1;
		REQUIRE(approxEqual(mat1, mat2));

		mat1.round();

		for (size_t i = 0; i < mat2.size(); ++i) {
			float val = mat2(i);
			mat2(i) = std::round(val);
		}

		REQUIRE(approxEqual(mat1, mat2));
	}

	SECTION("Rint")
	{
		mat1.rand(-1000, 1000);
		mat2 = mat1;
		REQUIRE(approxEqual(mat1, mat2));

		mat1.rint();

		for (size_t i = 0; i < mat2.size(); ++i) {
			float val = mat2(i);
			mat2(i) = std::rint(val);
		}

		REQUIRE(approxEqual(mat1, mat2));
	}

	Cumat::end();
}
