cmake_minimum_required(VERSION 3.16)
project(cuda_hashing CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(cuda_hashing main.cu hash_table.cu benchmark.cu)

target_include_directories(cuda_hashing PUBLIC ${CMAKE_SOURCE_DIR}/include)

set_target_properties(cuda_hashing PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
