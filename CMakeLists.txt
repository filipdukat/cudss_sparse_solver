cmake_minimum_required(VERSION 3.18)

project(cudss_solver LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

find_package(CUDAToolkit REQUIRED)


include_directories(
    /usr/include
)

link_directories(
    /usr/lib/x86_64-linux-gnu
)

add_executable(cudss_solver
    cudss_simple.cpp
    matrix_loader.cpp
)

target_link_libraries(cudss_solver
    cudss
    CUDA::cudart
    CUDA::cublas
)
