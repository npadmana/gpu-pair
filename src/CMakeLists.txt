cmake_minimum_required(VERSION 2.6)
PROJECT ( gpu-pair )

include_directories("$ENV{GSL_DIR}/include")
include_directories("$ENV{CUDA_DIR}/include")
link_directories("$ENV{GSL_DIR}/lib")


ADD_SUBDIRECTORY(global)
add_subdirectory(testing)
add_subdirectory(device)
add_subdirectory(gpu_testing)