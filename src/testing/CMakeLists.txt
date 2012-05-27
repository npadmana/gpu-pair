include_directories("${PROJECT_SOURCE_DIR}/global")

set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_C_COMPILER "gcc")

ADD_DEFINITIONS(-std=c++0x -O2)

add_executable(test_random.x test_random.cpp)
target_link_libraries(test_random.x pairs_global gsl gslcblas)

add_executable(test_particles1.x test_particles1.cpp)
target_link_libraries(test_particles1.x pairs_global gsl gslcblas)

add_executable(test_particles2.x test_particles2.cpp)
target_link_libraries(test_particles2.x pairs_global gsl gslcblas)

add_executable(timing_cpu_naive.x timing_cpu_naive.cpp)
target_link_libraries(timing_cpu_naive.x pairs_global gsl gslcblas)

add_executable(test_particles3.x test_particles3.cpp)
target_link_libraries(test_particles3.x pairs_global gsl gslcblas)

add_executable(test_particles_minmax.x test_particles_minmax.cpp)
target_link_libraries(test_particles_minmax.x pairs_global gsl gslcblas)

add_executable(test_particles_sort.x test_particles_sort.cpp)
target_link_libraries(test_particles_sort.x pairs_global gsl gslcblas)
    