#ifndef LIBCUMAT_OBJECTCOUNTER_H_
#define LIBCUMAT_OBJECTCOUNTER_H_

namespace Cumat
{

template<typename T>
class objectCounter
{
	protected:
	
	// Keeps track of total number of instantiated objects
	static unsigned int counter_;

	public:
	
	objectCounter(void) { ++counter_; }
	objectCounter(const objectCounter&) { ++counter_; }
	~objectCounter(void) { --counter_; }
	static unsigned int count(void) { return counter_; }
};

template<typename T>
unsigned int objectCounter<T>::counter_ = 0;

}

#endif
