cmake_minimum_required(VERSION 3.0)

project(libcumat VERSION 1.0)

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
# Note: CUDA include path is automatically included by cuda_add_executable
target_include_directories(libcumat INTERFACE
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
		$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
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
include (CMakePackageConfigHelpers)

set(LIBCUMAT_ROOT_DIR ${CMAKE_INSTALL_PREFIX})
set(LIBCUMAT_INCLUDE_DIRS ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR})
set(LIBCUMAT_LIBRARIES ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_curand_LIBRARY} ${CUDA_nvrtc_LIBRARY})
set(LIBCUMAT_DEFINITIONS "")
set(CMAKEPACKAGE_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR}/libcumat)

# Install Cumat targets
install(TARGETS libcumat
	EXPORT libcumatTargets
	LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
	ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# Setup libcumatConfig.cmake file
configure_package_config_file(
	${CMAKE_CURRENT_SOURCE_DIR}/cmake/libcumatConfig.cmake.in
	${CMAKE_CURRENT_BINARY_DIR}/libcumatConfig.cmake
	PATH_VARS LIBCUMAT_ROOT_DIR LIBCUMAT_INCLUDE_DIRS LIBCUMAT_LIBRARIES LIBCUMAT_DEFINITIONS
	INSTALL_DESTINATION ${CMAKEPACKAGE_INSTALL_DIR}
	NO_CHECK_REQUIRED_COMPONENTS_MACRO
)

# Setup libcumatConfigVersion.cmake file
write_basic_package_version_file(libcumatConfigVersion.cmake
	VERSION ${PACKAGE_VERSION}
	COMPATIBILITY SameMajorVersion
)

# Install include directory
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Install configuration target file
install(EXPORT libcumatTargets NAMESPACE libcumat:: DESTINATION ${CMAKEPACKAGE_INSTALL_DIR})

# Install config files into package folder
install(
	FILES
		${CMAKE_CURRENT_BINARY_DIR}/libcumatConfig.cmake
		${CMAKE_CURRENT_BINARY_DIR}/libcumatConfigVersion.cmake
	DESTINATION
		${CMAKEPACKAGE_INSTALL_DIR}
)

# Add uninstall target
add_custom_target(uninstall COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/libcumatUninstall.cmake)

# Allows use of ctest from build directory
enable_testing()

# By default, tests are left out of the all target
if(NOT LIBCUMAT_BUILD_TESTS)
	add_subdirectory(test EXCLUDE_FROM_ALL)
else()
	add_subdirectory(test)
endif()
