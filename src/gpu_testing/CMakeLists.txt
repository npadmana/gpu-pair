find_package(CUDA)
include(FindCUDA)

cuda_include_directories(${PROJECT_SOURCE_DIR}/device)
cuda_include_directories(${PROJECT_SOURCE_DIR}/global)

include_directories(${PROJECT_SOURCE_DIR}/device)
include_directories(${PROJECT_SOURCE_DIR}/global)


CUDA_ADD_EXECUTABLE(test_gpu_particles.x test_gpu_particles.cu OPTIONS -O2 -arch=sm_20)
target_link_libraries(test_gpu_particles.x pairs_global gsl gslcblas)

CUDA_ADD_EXECUTABLE(test_gpu_particles2.x test_gpu_particles2.cu OPTIONS -O2 -arch=sm_20)
target_link_libraries(test_gpu_particles2.x pairs_global pairs_device gsl gslcblas)

CUDA_ADD_EXECUTABLE(timing_naive.x timing_naive.cu OPTIONS -O2 -arch=sm_20)
target_link_libraries(timing_naive.x pairs_global pairs_device gsl gslcblas)

CUDA_ADD_EXECUTABLE(timing_shared.x timing_shared.cu OPTIONS -O2 -arch=sm_20)
target_link_libraries(timing_shared.x pairs_global pairs_device gsl gslcblas)

CUDA_ADD_EXECUTABLE(timing_shared_gpuonly.x timing_shared_gpuonly.cu OPTIONS -O2 -arch=sm_20)
target_link_libraries(timing_shared_gpuonly.x pairs_global pairs_device gsl gslcblas)