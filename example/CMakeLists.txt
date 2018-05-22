# This file builds example executables for library demonstration purposes

# Find all example files in the directory
file(GLOB libcumat_examples "*.cpp")

foreach(example ${libcumat_examples})

    # Get filename and extension
    get_filename_component(example_name ${example} NAME_WE)
    get_filename_component(example_ext ${example} EXT)

    # Allows cpp to be compiled as cuda files
    set_source_files_properties(${example_name}${example_ext} PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)

    # Add a cuda executable (this command automatically includes the CUDA include directory)
    cuda_add_executable(${example_name} ${example_name}${example_ext})

    # Link with the libcumat package
    # Note: For personal projects, you'll have to first install libcumat and then find the libcumat package through find_package
    target_link_libraries(${example_name} libcumat)

endforeach()
