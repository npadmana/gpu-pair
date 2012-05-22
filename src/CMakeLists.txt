cmake_minimum_required(VERSION 2.6)
PROJECT ( gpu-pair )

include_directories("$ENV{GSL_DIR}/include")
include_directories("/usr/local/cuda/include")
link_directories("$ENV{GSL_DIR}/lib")


ADD_SUBDIRECTORY(global)
add_subdirectory(testing)