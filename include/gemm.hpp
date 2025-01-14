#ifdef AMD
#include <hip/hip_runtime.h>
#endif

#ifdef NVIDIA
#include <cuda_runtime.h>
#endif


__global__ void dgemm(const double *A, const double *B, double *D) {
    #ifdef AMD
    printf("Running on AMD GPU \n");
    #endif

    #ifdef NVIDIA
    printf("Running on NVIDIA GPU \n");
    #endif

}