/// GEMM D = \alpha A @ B + \beta * C

#ifdef AMD
#include <hip/hip_runtime.h>
#endif

#ifdef NVIDIA
#include <cuda_runtime.h>
#include "gemm_cuda.hpp"
#include <cublas.h>
#include <cutensor.h>
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
        cublasHandle_t handle;
        cublasStatus_t status = cublasCreate_v2(&handle);

    float milliseconds = run_kernel_with_optional_timing_cuda([ = ]() {
        // sgemm_gmem_coalesced<128> <<< grid, block>>>(M, N, K, alpha, A, B, beta, C);
        // sgemm_simple<<< grid, block>>>(M, N, K, alpha, A, B, beta, C);

        // cublasSgemm_v2(
        //             handle,
        //             CUBLAS_OP_T, CUBLAS_OP_T,  // Transpose both matrices for row-major
        //             N, M, K,                  // Adjust dimensions for column-major order
        //             &alpha,                   // Scaling factor for A * B
        //             B, K,                     // B matrix (row-major -> transposed)
        //             A, M,                     // A matrix (row-major -> transposed)
        //             &beta,                    // Scaling factor for C
        //             C, N);                    // C matrix (row-major output)

        // cublasSgemm_v2(
        //     handle,
        //     CUBLAS_OP_N, CUBLAS_OP_N,
        //     M, N, K,
        //     &alpha,
        //     A, M,
        //     B, K,
        //     &beta,
        //     C, N
        // );

    }, timed);



    std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;
    auto flops = count_flops_gemm(M, N, K);
    auto [read, write] = count_memory_gemm<float>(M, N, K);
    double gflops = (double)flops / 1e9; // Convert to GFLOPs

    printf("FLOPs: %f GFLOPs \n", gflops);
    printf("Reads: %f MB \n", static_cast<double>(read) / (double)(1024 * 1024));
    printf("Writes: %f MB \n", static_cast<double>(write) / (double)(1024 * 1024));
    printf("Throughput: %f GFLOPS \n", gflops /( milliseconds / 1000));
    printf("Throughput: %f GB/s \n", (static_cast<double>(read) / (double(1024 * 1024 * 1024))) /( milliseconds / 1000));

#endif
}