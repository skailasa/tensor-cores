#include <stdexcept>
#include <utils.hpp>
#include <kernels.hpp>

#ifdef NVIDIA
#include <cuda_runtime.h>
#include <cublas.h>
#include <cutensor.h>
#endif


/// @brief  Call cubLAS with single precision inputs
void runCublasF32(cublasHandle_t handle, int M, int N, int K, float alpha, float *A, float *B, float beta, float*C) {
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_32F,
        N, A, CUDA_R_32F, K, &beta, C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}


/// @brief  Call cubLAS with single precision inputs casted down to BF16 for the actual mul
void runCublasB16(cublasHandle_t handle, int M, int N, int K, float alpha, float *A, float *B, float beta, float*C) {
    // CUBLAS uses col-major by default, we use row-major by default
    // TODO: configure to handle both (B^T A^T)^T = A B
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_32F,
        N, A, CUDA_R_32F, K, &beta, C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F_FAST_16BF, CUBLAS_GEMM_DEFAULT);
}

/// @brief  Call cubLAS with single precision inputs casted to TF32 for the actual mul
void runCublasT32(cublasHandle_t handle, int M, int N, int K, float alpha, float *A, float *B, float beta, float*C) {
    // CUBLAS uses col-major by default, we use row-major by default
    // TODO: configure to handle both (B^T A^T)^T = A B
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_32F,
        N, A, CUDA_R_32F, K, &beta, C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT);
}

/// @brief  Call cubLAS with single precision inputs casted down to F16 for the actual mul
void runCublasF16(cublasHandle_t handle, int M, int N, int K, float alpha, float *A, float *B, float beta, float*C) {
    // CUBLAS uses col-major by default, we use row-major by default
    // TODO: configure to handle both (B^T A^T)^T = A B
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_32F,
        N, A, CUDA_R_32F, K, &beta, C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT);
}

void runSgemmCpu(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {

    // Perform matrix multiplication
    for (int i = 0; i < M; i++) {         // Iterate over rows of A and C
        for (int j = 0; j < N; j++) {     // Iterate over columns of B and C
            for (int k = 0; k < K; k++) { // Sum over the inner K-dimension
                C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
}

void runSGemmNaive(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    dim3 gridDim(ceil_div(M, 32), ceil_div(N, 32));
    dim3 blockDim(32, 32);
    sgemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void runSgemmGmemCoalescing(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    dim3 gridDim(ceil_div(M, 32), ceil_div(N, 32));
    dim3 blockDim(32 * 32);
    const uint BLOCKSIZE = 32;
    sgemm_gmem_coalescing<BLOCKSIZE><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}


float runKernel32(int kernel_number, int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {

    cublasStatus_t status; // cuBLAS functions status
    cublasHandle_t handle; // cublas context
    status = cublasCreate_v2(&handle); // initialize CUBLAS context
    float time;

    switch (kernel_number) {

        // Default cuBLAS call in single precision
        case 0:
            time = run_kernel_with_optional_timing( [ = ]()  {
                runCublasF32(handle, M, N, K, alpha, A, B, beta, C);
            }, true);

            return time;
        break;

        // brain float precision cublas call
        case 1:
            time = run_kernel_with_optional_timing( [ = ]()  {
                runCublasB16(handle, M, N, K, alpha, A, B, beta, C);
            }, true);
        break;

        // tensor float precision cublas call
        case 2:
            time = run_kernel_with_optional_timing( [ = ]()  {
                runCublasT32(handle, M, N, K, alpha, A, B, beta, C);
            }, true);
        break;

        // half precision cublas call
        case 3:
            time = run_kernel_with_optional_timing( [ = ]()  {
                runCublasF16(handle, M, N, K, alpha, A, B, beta, C);
            }, true);
        break;

        // single precision cublas call
        case 4:
            time = run_kernel_with_optional_timing( [ = ]()  {
                runSGemmNaive(M, N, K, alpha, A, B, beta, C);
            }, true);

        break;

        // shared memory coalesced reads
        case 5:
            time = run_kernel_with_optional_timing( [ = ]() {
                runSgemmGmemCoalescing(M, N, K, alpha, A, B, beta, C);
            }, true);
        break;

        default:
            throw std::invalid_argument("Unknown kernel number");
    }

    return time;
}