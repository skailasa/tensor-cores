#include <stdexcept>
#include <utils.hpp>
#include <kernels.hpp>

#ifdef NVIDIA
#include <cuda_runtime.h>
#include <cublas.h>
#include <cutensor.h>
#endif


// Type alias for GEMM kernels i.e. pointers to kernel functions
// equivalent to typedef void (*my_kernel)(arg1, arg2,...) - a raw pointer to a function
// can point to any function with this/that signature
using KernelPtr = void(*)(int, int, int, float, const float*, const float*, float, float*);

/// @brief  Call cubLAS with single precision inputs
void runCublasF32(cublasHandle_t handle, Layout layout, int M, int N, int K, float alpha, float *A, float *B, float beta, float*C) {

    if (layout == Layout::RowMajor) {
        cublasSgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K, &beta, C, N);
    } else if (layout == Layout::ColumnMajor) {
        cublasSgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, A, M, B, K, &beta, C, M);
    }
}


/// @brief  Call cubLAS with single precision inputs casted down to BF16 for the actual mul
void runCublasB16(cublasHandle_t handle, Layout layout, int M, int N, int K, float alpha, float *A, float *B, float beta, float*C) {
    if (layout == Layout::RowMajor) {
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_32F,
        N, A, CUDA_R_32F, K, &beta, C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F_FAST_16BF, CUBLAS_GEMM_DEFAULT);
    } else if (layout == Layout::ColumnMajor) {
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, A, CUDA_R_32F,
        M, B, CUDA_R_32F, K, &beta, C, CUDA_R_32F, M, CUBLAS_COMPUTE_32F_FAST_16BF, CUBLAS_GEMM_DEFAULT);
    }
}

/// @brief  Call cubLAS with single precision inputs casted to TF32 for the actual mul
void runCublasT32(cublasHandle_t handle, Layout layout, int M, int N, int K, float alpha, float *A, float *B, float beta, float*C) {
    if (layout == Layout::RowMajor) {
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_32F,
        N, A, CUDA_R_32F, K, &beta, C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT);
    } else if (layout == Layout::ColumnMajor) {
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, A, CUDA_R_32F,
        M, B, CUDA_R_32F, K, &beta, C, CUDA_R_32F, M, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT);
    }
}

/// @brief  Call cubLAS with single precision inputs casted down to F16 for the actual mul
void runCublasF16(cublasHandle_t handle, Layout layout, int M, int N, int K, float alpha, float *A, float *B, float beta, float*C) {
    if (layout == Layout::RowMajor) {
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_32F,
        N, A, CUDA_R_32F, K, &beta, C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT);
    } else if (layout == Layout::ColumnMajor) {
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, A, CUDA_R_32F,
        M, B, CUDA_R_32F, K, &beta, C, CUDA_R_32F, M, CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT);
    }
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

void runSGemmNaive(Layout layout, cudaFuncCache cache_configuration,  int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    dim3 gridDim(ceil_div(M, 32), ceil_div(N, 32));
    dim3 blockDim(32, 32);

    if (layout == Layout::RowMajor) {
        KernelPtr kernel = sgemm_naive_row_major;
        cudaFuncSetCacheConfig(kernel, cache_configuration);
        kernel<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);

    } else if (layout == Layout::ColumnMajor) {
        KernelPtr kernel = sgemm_naive_column_major;
        cudaFuncSetCacheConfig(kernel, cache_configuration);
        kernel<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
    }
}


void runSgemmSharedMemCacheBlocking(Layout layout, cudaFuncCache cache_configuration, int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {

    const uint BLOCKSIZE = 32;
    if (layout == Layout::RowMajor) {
        dim3 blockDim(BLOCKSIZE, BLOCKSIZE);
        dim3 gridDim(ceil_div(M, BLOCKSIZE), ceil_div(N, BLOCKSIZE));
        KernelPtr kernel = sgemm_smem_cache_blocking_row_major<BLOCKSIZE>;
        cudaFuncSetCacheConfig(kernel, cache_configuration);
        kernel<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);

    } else if (layout == Layout::ColumnMajor) {
        dim3 blockDim(BLOCKSIZE, BLOCKSIZE);
        dim3 gridDim(ceil_div(N, BLOCKSIZE), ceil_div(M, BLOCKSIZE));
        KernelPtr kernel = sgemm_smem_cache_blocking_column_major<BLOCKSIZE>;
        cudaFuncSetCacheConfig(kernel, cache_configuration);
        kernel<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
    }
}

void runSgemm1dBlockTiling(Layout layout, cudaFuncCache cache_configuration, int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {

    const uint BM = 64;
    const uint BN = 64;
    const uint BK = 16;
    const uint TM = 4;

    static_assert(BM % TM == 0, "BM must be divisible by TM");

    if (layout == Layout::RowMajor) {
        dim3 gridDim(ceil_div(M, BM), ceil_div(N, BN)); // same as in shared mem cache blocking, but with tunable parameters
        dim3 blockDim((BM * BN)/TM); // each thread in block responsible for TM values of output
        // and total output size of each block is still BN * BM
        KernelPtr kernel = sgemm_smem_1d_blocktiling_row_major<BM, BN, BK, TM>;
        cudaFuncSetCacheConfig(kernel, cache_configuration);
        kernel<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
    } else if (layout == Layout::ColumnMajor) {
        dim3 gridDim(ceil_div(M, BM), ceil_div(N, BN)); // same as in shared mem cache blocking, but with tunable parameters
        dim3 blockDim((BM * BN)/TM); // each thread in block responsible for TM values of output
        KernelPtr kernel = sgemm_smem_1d_blocktiling_column_major<BM, BN, BK, TM>;
        cudaFuncSetCacheConfig(kernel, cache_configuration);
        kernel<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
    }
}


void runSgemm2dBlockTiling(Layout layout, cudaFuncCache cache_configuration, int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {

    const uint BM = 64;
    const uint BN = 64;
    const uint BK = 16;
    const uint TM = 4;
    const uint TN = 4;
    static_assert((BM % TM == 0) && (BN % TN == 0), "BM must be divisible by TM and BN must be divisible by TN");

    if (layout == Layout::RowMajor) {
        dim3 gridDim(ceil_div(M, BM), ceil_div(N, BN)); // same as in shared mem cache blocking, but with tunable parameters
        dim3 blockDim((BM * BN) / ((TM * TN)));
        KernelPtr kernel = sgemm_smem_2d_blocktiling_row_major<BM, BN, BK, TM, TN>;
        cudaFuncSetCacheConfig(kernel, cache_configuration);
        kernel<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);

    } else if (layout == Layout::ColumnMajor) {
        // dim3 gridDim(ceil_div(N, BN), ceil_div(M, BM)); // same as in shared mem cache blocking, but with tunable parameters
        // KernelPtr kernel = sgemm_smem_2d_blocktiling_column_major<BM, BN, BK, TM, TN>;
        // cudaFuncSetCacheConfig(kernel, cache_configuration);
        // kernel<<(M, N, K, alpha, A, B, beta, C);
    }

}


float runKernel32(int kernel_number, Layout layout, cudaFuncCache cache_configuration, int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {

    cublasHandle_t handle; // cublas context
    auto _status = cublasCreate_v2(&handle); // initialize CUBLAS context
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

    // cuBLAS with TF32
    case 1:
        // Warmup call
        runCublasT32(handle, layout, M, N, K, alpha, A, B, beta, C);
        time = run_kernel_with_optional_timing( [ = ]()  {
            runCublasT32(handle, layout, M, N, K, alpha, A, B, beta, C);
        }, true);

        return time;
        break;

    // cuBLAS with BF16
    case 2:
        // Warmup call
        runCublasB16(handle, layout, M, N, K, alpha, A, B, beta, C);
        time = run_kernel_with_optional_timing( [ = ]()  {
            runCublasB16(handle, layout, M, N, K, alpha, A, B, beta, C);
        }, true);

        return time;
        break;

    // cuBLAS with half precision
    case 3:
        // Warmup call
        runCublasF16(handle, layout, M, N, K, alpha, A, B, beta, C);
        time = run_kernel_with_optional_timing( [ = ]()  {
            runCublasF16(handle, layout, M, N, K, alpha, A, B, beta, C);
        }, true);

        return time;
        break;

    // single naive kernel
    case 4:
        // Warmup call
        time = run_kernel_with_optional_timing( [ = ]()  {
            runSGemmNaive(layout, cache_configuration, M, N, K, alpha, A, B, beta, C);
        }, true);

        break;

    // single naive kernel
    case 7:
        time = run_kernel_with_optional_timing( [ = ]()  {
            runSgemmSharedMemCacheBlocking(layout, cache_configuration, M, N, K, alpha, A, B, beta, C);
        }, true);

        break;

    case 8:
        time = run_kernel_with_optional_timing( [ = ]()  {
            runSgemm1dBlockTiling(layout, cache_configuration, M, N, K, alpha, A, B, beta, C);
        }, true);

        break;

    case 9:
        time = run_kernel_with_optional_timing( [ = ]()  {
            runSgemm1dBlockTiling(layout, cache_configuration, M, N, K, alpha, A, B, beta, C);
        }, true);

        break;

    case 10:
        time = run_kernel_with_optional_timing( [ = ]()  {
            runSgemm2dBlockTiling(layout, cache_configuration, M, N, K, alpha, A, B, beta, C);
        }, true);

        break;



    default:
        throw std::invalid_argument("Unknown kernel number");
    }

    return time;
}