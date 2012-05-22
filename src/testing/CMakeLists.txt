include_directories("${PROJECT_SOURCE_DIR}/global")

set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_C_COMPILER "gcc")

ADD_DEFINITIONS(-std=c++0x -O2)

add_executable(test_random.x test_random.cpp)
target_link_libraries(test_random.x pairs_global gsl)

add_executable(test_particles1.x test_particles1.cpp)
target_link_libraries(test_particles1.x pairs_global gsl)
