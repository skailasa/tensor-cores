#include <stdexcept>
#include <utils.hpp>
#include <kernels.hpp>

#ifdef NVIDIA
#include <cuda_runtime.h>
#include <cublas.h>
#include <cutensor.h>
#endif


/// @brief  Call cubLAS with single precision inputs
void runCublasF32(cublasHandle_t handle, Layout layout, int M, int N, int K, float alpha, float *A, float *B, float beta, float*C) {

    if (layout == Layout::RowMajor) {
        cublasSgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K, &beta, C, N);
    } else if (layout == Layout::ColumnMajor) {
        cublasSgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, A, M, B, K, &beta, C, M);
    }
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

void runSgemmCpu(Layout layout, int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {

    // Perform matrix multiplication
    if (layout == Layout::RowMajor) {
        // C[i * N + j] = alpha * A[i * K + k] * B[k * N + j] + beta * C[i * N + j]
        for (int row = 0; row < M; row++) {
            for (int col = 0; col < N; col++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    sum += A[row * K + k] * B[k * N + col];
                }
                C[row * N + col] = alpha * sum + beta * C[row * N + col];
            }
        }
    } else if (layout == Layout::ColumnMajor) {
        // C[i + j * M] = alpha * sum(A[i + k * M] * B[k + j * K]) + beta * C[i + j * M]
        for (int col = 0; col < N; col++) {        // column of C and B
            for (int row = 0; row < M; row++) {    // row of C and A
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    sum += A[k * M + row] * B[col * K + k];
                }
                C[col * M + row] = alpha * sum + beta * C[col * M + row];
            }
        }
    }
}

void runSGemmNaive(Layout layout, int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    dim3 gridDim(ceil_div(M, 32), ceil_div(N, 32));
    dim3 blockDim(32, 32);

    if (layout == Layout::RowMajor) {
        sgemm_naive_row_major<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
    } else if (layout == Layout::ColumnMajor) {
        sgemm_naive_column_major<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
    }
}


void runSgemmGmemCoalescing(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    dim3 gridDim(ceil_div(M, 32), ceil_div(N, 32));
    dim3 blockDim(32 * 32);
    const uint BLOCKSIZE = 32;
    sgemm_gmem_coalescing<BLOCKSIZE><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}


float runKernel32(int kernel_number, Layout layout, int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {

    cublasStatus_t status; // cuBLAS functions status
    cublasHandle_t handle; // cublas context
    status = cublasCreate_v2(&handle); // initialize CUBLAS context
    float time;

    switch (kernel_number) {

        // Default cuBLAS call in single precision
        case 0:
            // Warmup call
            runCublasF32(handle, layout, M, N, K, alpha, A, B, beta, C);
            time = run_kernel_with_optional_timing( [ = ]()  {
                runCublasF32(handle, layout, M, N, K, alpha, A, B, beta, C);
            }, true);

            return time;
        break;

        // single naive kernel
        case 1:
            runSGemmNaive(layout, M, N, K, alpha, A, B, beta, C);
            time = run_kernel_with_optional_timing( [ = ]()  {
                runSGemmNaive(layout, M, N, K, alpha, A, B, beta, C);
            }, true);

        break;

        // // brain float precision cublas call
        // case 1:
        //     time = run_kernel_with_optional_timing( [ = ]()  {
        //         runCublasB16(handle, M, N, K, alpha, A, B, beta, C);
        //     }, true);
        // break;

        // // tensor float precision cublas call
        // case 2:
        //     time = run_kernel_with_optional_timing( [ = ]()  {
        //         runCublasT32(handle, M, N, K, alpha, A, B, beta, C);
        //     }, true);
        // break;

        // // half precision cublas call
        // case 3:
        //     time = run_kernel_with_optional_timing( [ = ]()  {
        //         runCublasF16(handle, M, N, K, alpha, A, B, beta, C);
        //     }, true);
        // break;



        // // shared memory coalesced reads
        // case 1:
        //     time = run_kernel_with_optional_timing( [ = ]() {
        //         runSgemmGmemCoalescing(M, N, K, alpha, A, B, beta, C);
        //     }, true);
        // break;

        default:
            throw std::invalid_argument("Unknown kernel number");
    }

    return time;
}