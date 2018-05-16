project(libcumat)

cmake_minimum_required(VERSION 3.0)

# Guard against in-source builds
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
	message(FATAL_ERROR "In-source builds not allowed. Please make a new directory and run CMake from there. You may need to remove CMakeCache.txt.")
endif()

# Set CUDA related variables
find_package(CUDA REQUIRED)

# Set Cmake modules location
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/)

# Set libcumat as an interface library
add_library(libcumat INTERFACE)

# Sets variables for the CUDA nvrtc library
include(FindNVRTC)

# Add necessary include directories
target_include_directories(libcumat INTERFACE
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
		$<INSTALL_INTERFACE:include>
		${CUDA_INCLUDE_DIRS}
)

# Link necessary CUDA libraries
target_link_libraries(libcumat INTERFACE
	${CUDA_LIBRARIES}
	${CUDA_CUDA_LIBRARY}
	${CUDA_CUBLAS_LIBRARIES}
	${CUDA_curand_LIBRARY}
	${CUDA_nvrtc_LIBRARY}
)

include(GNUInstallDirs)

# Install Cumat targets
install(TARGETS libcumat
	EXPORT libcumatTargets
	LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
	ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# Install include directory
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Install configuration file
install(EXPORT libcumatTargets
	FILE libcumatConfig.cmake
	NAMESPACE libcumat::
	DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/libcumat
)
