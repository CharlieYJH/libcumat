#ifndef CPUMATRIX_HPP_
#define CPUMATRIX_HPP_

#include <iostream>
#include <iomanip>
#include <vector>
#include <assert.h>
#include <cstdlib>
#include <algorithm>

template<typename T>
struct CPUMatrix
{
	size_t rows;
	size_t cols;
	std::vector<T> data;
	CPUMatrix(size_t r, size_t c) : rows(r), cols(c), data(r * c, 0) {}
	void print(void);
	void rand(T min = -1, T max = 1);
};

template<typename T>
void CPUMatrix<T>::print(void)
{
	for (size_t i = 0; i < rows; ++i) {
		for (size_t j = 0; j < cols; ++j)
			std::cout << std::setw(8) << data[i * cols + j] << " ";
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

template<typename T>
void CPUMatrix<T>::rand(T min, T max)
{
	if (min > max)
		std::swap(min, max);

	for (size_t i = 0; i < data.size(); ++i) {
		data[i] = (double)std::rand() / (double)RAND_MAX;
		data[i] = data[i] * (max - min) + min;
	}
}

template<typename T>
void CPUMatrixMultiply(CPUMatrix<T> &left, CPUMatrix<T> &right, CPUMatrix<T> &result)
{
	assert(left.cols == right.rows && result.rows == left.rows && result.cols == right.cols);
	
	for (size_t i = 0; i < result.rows; ++i) {
		for (size_t j = 0; j < result.cols; ++j) {
			
			T temp_sum = 0;

			for (size_t k = 0; k < left.cols; ++k)
				temp_sum += left.data[i * left.cols + k] * right.data[k * right.cols + j];

			result.data[i * result.cols + j] = temp_sum;
		}
	}
}

#endif
