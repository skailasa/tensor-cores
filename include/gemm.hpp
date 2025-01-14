/// GEMM D = \alpha A @ B + \beta * C

#ifdef AMD
#include <hip/hip_runtime.h>
#endif

#ifdef NVIDIA
#include <cuda_runtime.h>
#include "gemm_cuda.hpp"
#endif



__global__ void dgemm(int M, int N, int K, double alpha, const double *A, const double *B,
                      double beta, double *C) {
#ifdef AMD
    printf("Running on AMD GPU \n");
#endif

#ifdef NVIDIA
    printf("Running on NVIDIA GPU \n");
#endif
}

void sgemm(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C,
           dim3 grid, dim3 block, bool timed) {
#ifdef AMD
    printf("Running on AMD GPU \n");
#endif

#ifdef NVIDIA
    printf("Running on NVIDIA GPU \n");
    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);

    // cudaEventRecord(start, 0);

    // sgemm_simple<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);

    // cudaEventRecord(stop, 0);
    // cudaEventSynchronize(stop);
    // float milliseconds = 0;
    // cudaEventElapsedTime(&milliseconds, start, stop);
    // std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;
    // cudaEventDestroy(start);
    // cudaEventDestroy(stop);

    run_kernel_with_optional_timing_cuda([ = ]() {
        sgemm_simple <<< grid, block>>>(M, N, K, alpha, A, B, beta, C);
    }, timed);
#endif
}