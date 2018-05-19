#include <vector>
#include "catch.hpp"
#include "libcumat.h"

TEST_CASE("Counting instances of matrix object")
{
	unsigned int counter = 0;

	// Instantiating on stack
	Cumat::Matrixf mat1;
	++counter;
	REQUIRE(Cumat::objectCounter<Cumat::CudaHandler>::count() == counter);

	// Instantiating on heap
	Cumat::Matrixf *mat2 = new Cumat::Matrixf(20, 30);
	++counter;
	REQUIRE(Cumat::objectCounter<Cumat::CudaHandler>::count() == counter);

	// Free heap variable
	delete mat2;
	--counter;
	REQUIRE(Cumat::objectCounter<Cumat::CudaHandler>::count() == counter);

	// Assignment constructor doesn't increase count twice
	Cumat::Matrixf mat3 = Cumat::Matrixf::random(50, 60);
	++counter;
	REQUIRE(Cumat::objectCounter<Cumat::CudaHandler>::count() == counter);

	// Instantiating in scope
	for (size_t i = 0; i < 300; ++i) {
		std::cout << "Loop start" << std::endl;
		Cumat::Matrixd temp = Cumat::Matrixf::random(40, 40);
		++counter;
		REQUIRE(Cumat::objectCounter<Cumat::CudaHandler>::count() == counter);
		--counter;
		std::cout << "Loop end" << std::endl;
	}

	// All objects in scope should be automatically destroyed at end of scope
	REQUIRE(Cumat::objectCounter<Cumat::CudaHandler>::count() == counter);

	Cumat::Matrixd *mat4;

	// Instantiating on heap inside scope
	if (true) {
		mat4 = new Cumat::Matrixd(100, 100);
		++counter;
	}

	// Object on heap should remain
	REQUIRE(Cumat::objectCounter<Cumat::CudaHandler>::count() == counter);

	// Vector of matrices on stack
	std::vector<Cumat::Matrixd> matrix_vec_d(300);
	counter += matrix_vec_d.size();

	// All matrices in vector should be accounted for
	REQUIRE(Cumat::objectCounter<Cumat::CudaHandler>::count() == counter);

	// Instantiate vector of matrices in scope
	if (true) {
		std::vector<Cumat::Matrixf> matrix_vec_f(203);
		counter += matrix_vec_f.size();
		REQUIRE(Cumat::objectCounter<Cumat::CudaHandler>::count() == counter);
		counter -= matrix_vec_f.size();
	}

	// All matrices in vector should be deleted once scope ends
	REQUIRE(Cumat::objectCounter<Cumat::CudaHandler>::count() == counter);

	// Free matrix allocated on heap earlier
	delete mat4;
	--counter;
	REQUIRE(Cumat::objectCounter<Cumat::CudaHandler>::count() == counter);

	// Resize previous matrix
	size_t prev_size = matrix_vec_d.size();
	matrix_vec_d.resize(19);
	counter -= (prev_size - matrix_vec_d.size());
	REQUIRE(Cumat::objectCounter<Cumat::CudaHandler>::count() == counter);

	// References don't add onto counter
	Cumat::Matrixf &ref_mat = mat1;
	REQUIRE(Cumat::objectCounter<Cumat::CudaHandler>::count() == counter);

	// Operations don't increase count
	mat1 = mat3 * 2;
	mat1.transpose();
	REQUIRE(Cumat::objectCounter<Cumat::CudaHandler>::count() == counter);

	// Matrix multiplication doesn't increase count
	mat1 = mmul(mat3, transpose(mat3));
	REQUIRE(Cumat::objectCounter<Cumat::CudaHandler>::count() == counter);
}
