set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_C_COMPILER "gcc")

ADD_DEFINITIONS(-std=c++0x -O2)

ADD_LIBRARY(pairs_global SHARED Particles.h NPVector.h cpu_utils.cpp Hist.cpp CPU_PairCounts.cpp)
TARGET_LINK_LIBRARIES(pairs_global gsl gslcblas)
