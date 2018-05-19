# This file builds the example executable for library demonstration purposes

# This line allows cpp files to be compiled as cuda files
set_source_files_properties(example.cpp PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)

# Add a cuda executable (this command automatically includes the CUDA include directory)
cuda_add_executable(example example.cpp)

# Link with the libcumat package
# Note: For personal projects, you'll have to first install libcumat and then find the libcumat package through find_package
target_link_libraries(example libcumat)
