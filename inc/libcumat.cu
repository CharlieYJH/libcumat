#include "libcumat.h"

//----------------------------------------------
// Private methods
//----------------------------------------------

template<typename T>
T cumat<T>::generateRandom(const T &min, const T &max)
{
	assert(max > min);
	T random = ((T) std::rand()) / (T) RAND_MAX;
	T diff = max - min;
	T r = random * diff;
	return min + r;
}

//----------------------------------------------
// Public methods
//----------------------------------------------

template<typename T>
cumat<T>::cumat(size_t rows, size_t cols):
	m_rows(rows),
	m_cols(cols)
{
	m_data.resize(rows * cols);
}

template<typename T>
cumat<T>::cumat(void):
	m_rows(0),
	m_cols(0)
{}

template<typename T>
size_t cumat<T>::rows(void) const
{
	return m_rows;
}

template<typename T>
size_t cumat<T>::cols(void) const
{
	return m_cols;
}

template<typename T>
size_t cumat<T>::size(void) const
{
	return m_rows * m_cols;
}

template<typename T>
T cumat<T>::get(const size_t &row, const size_t &col) const
{
	assert(row < m_rows && col < m_cols);
	return m_data[row * m_cols + col];
}

template<typename T>
void cumat<T>::set(const size_t &row, const size_t &col, const T &val)
{
	assert(row < m_rows && col < m_cols);
	m_data[row * m_cols + col] = val;
}

template<typename T>
void cumat<T>::fill(const T &val)
{
	thrust::fill(m_data.begin(), m_data.end(), val);
}

template<typename T>
void cumat<T>::zero(void)
{
	cumat<T>::fill(0);
}

template<typename T>
void cumat<T>::rand()
{
	thrust::host_vector<T> vec(m_data.size());

	for (int i = 0; i < vec.size(); i++)
		vec[i] = cumat<T>::generateRandom(-1, 1);

	m_data = vec;
}

template<typename T>
void cumat<T>::rand(const T &min, const T &max)
{
	thrust::host_vector<T> vec(m_data.size());

	for (int i = 0; i < vec.size(); i++)
		vec[i] = cumat<T>::generateRandom(min, max);

	m_data = vec;
}

//----------------------------------------------
// Operator Overloads
//----------------------------------------------

// Template explicit instantiation
template class cumat<int>;
template class cumat<float>;
template class cumat<double>;
