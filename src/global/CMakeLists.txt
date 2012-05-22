set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_C_COMPILER "gcc")

include_directories("$ENV{GSL_DIR}/include")
link_directories("$ENV{GSL_DIR}/lib")
ADD_DEFINITIONS(-std=c++0x -O2)

ADD_LIBRARY(pairs_global SHARED Particles.h cpu_utils.h cpu_utils.cpp)
TARGET_LINK_LIBRARIES(pairs_global gsl)
