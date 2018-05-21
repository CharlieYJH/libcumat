# This file adds all the tests in the test folder to ctest
# Each test file compiles into its own executable

# Set include path for Catch header
set(CATCH_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/catch)

# Add Catch as an interface library and set its include directory
add_library(Catch INTERFACE)
target_include_directories(Catch INTERFACE ${CATCH_INCLUDE_DIRS})

# All tests will statically link with the Catch entry point
add_library(catch_main STATIC catch_main.cpp)
target_link_libraries(catch_main Catch)

file(GLOB libcumat_test_files "*.cpp")

foreach(file ${libcumat_test_files})

    # Get the filename and extension
    get_filename_component(test_name ${file} NAME_WE)
    get_filename_component(test_ext ${file} EXT)

    # Don't add a test for the Catch entry point file
    if(NOT test_name STREQUAL catch_main)

        # This allows cpp to be compiled as cuda files
        set_source_files_properties(${test_name}${test_ext} PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)

        # Add executable and link necessary libraries
        cuda_add_executable(${test_name} ${test_name}${test_ext})
        target_link_libraries(${test_name} catch_main libcumat)

        # Add test to ctest
        add_test(${test_name} ${test_name})

    endif()
endforeach()
