#ifndef LIBCUMAT_TYPESTRINGS_H_
#define LIBCUMAT_TYPESTRINGS_H_

#include <string>

namespace Cumat
{

// Workaround for nvcc behaviour where static const variables in a template class/struct wasn't being initialized
template<typename T>
struct TypeString {};

template<>
struct TypeString<double> { static const std::string type; };
const std::string TypeString<double>::type = "double";

template<>
struct TypeString<float> { static const std::string type; };
const std::string TypeString<float>::type = "float";

template<>
struct TypeString<int> { static const std::string type; };
const std::string TypeString<int>::type = "int";

}

#endif
