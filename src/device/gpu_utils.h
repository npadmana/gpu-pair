/*
 * gpu_utils.h
 *
 *  Created on: May 22, 2012
 *      Author: npadmana
 */

#ifndef GPU_UTILS_H_
#define GPU_UTILS_H_

#include <cuda_runtime_api.h>
#include <thrust/system/cuda_error.h> // It looks like this may have changed for Thrust v1.6
#include <thrust/system_error.h>
#include <string>
#include <Particles.h>


/* The code below is shamelessly cribbed from the Thrust examples
 * All copyrights are theirs
 */

void cuda_safe_call(cudaError_t error, const std::string& message = "")
{
  if(error)
    throw thrust::system_error(error, thrust::cuda_category(), message);
}

class GPUclock
{
  cudaEvent_t start;
  cudaEvent_t end;

	public :
  GPUclock(void)
  {
    cuda_safe_call(cudaEventCreate(&start));
    cuda_safe_call(cudaEventCreate(&end));
    restart();
  }

  ~GPUclock(void)
  {
    cuda_safe_call(cudaEventDestroy(start));
    cuda_safe_call(cudaEventDestroy(end));
  }

  void restart(void)
  {
    cuda_safe_call(cudaEventRecord(start, 0));
  }

  double elapsed(void)
  {
    cuda_safe_call(cudaEventRecord(end, 0));
    cuda_safe_call(cudaEventSynchronize(end));

    float ms_elapsed;
    cuda_safe_call(cudaEventElapsedTime(&ms_elapsed, start, end));
    return ms_elapsed / 1e3;
  }

  double epsilon(void)
  {
    return 0.5e-6;
  }
};



#endif /* GPU_UTILS_H_ */
