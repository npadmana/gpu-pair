find_package(CUDA)
include(FindCUDA)

cuda_include_directories(${PROJECT_SOURCE_DIR}/device)
cuda_include_directories(${PROJECT_SOURCE_DIR}/global)

include_directories(${PROJECT_SOURCE_DIR}/device)
include_directories(${PROJECT_SOURCE_DIR}/global)


CUDA_ADD_LIBRARY(pairs_device SHARED GPU_PairCounts.cu OPTIONS -O2 -arch=sm_20 -prec-sqrt=false)
target_link_libraries(pairs_device)