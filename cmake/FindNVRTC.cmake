# - Finds the CUDA NVRTC library
#
# This will define the following variables:
# CUDA_nvrtc_LIBRARY - The nvrtc library
# CUDA_nvrtc_FOUND - Whether the library was found

find_package(PkgConfig)

find_library(CUDA_nvrtc_LIBRARY
	NAMES libnvrtc nvrtc
	HINTS "${CUDA_TOOLKIT_ROOT_DIR}/lib64" "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64" "${CUDA_nvrtc_LIBRARY_DIR}"
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CUDA_nvrtc
	FOUND_VAR CUDA_nvrtc_FOUND
	REQUIRED_VARS CUDA_nvrtc_LIBRARY
)

mark_as_advanced(CUDA_nvrtc_LIBRARY)

if(NOT CUDA_nvrtc_FOUND)
	message(FATAL_ERROR "CUDA NVRTC library not found. Specify CUDA_nvrtc_LIBRARY_DIR where the library is located.")
endif()
