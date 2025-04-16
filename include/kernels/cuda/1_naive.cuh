#pragma once

/// @brief  Simplest GEMM kernel, each thread is responsible for an entry in the output matrix. Expect matrix data in
/// row major order
/// @param M
/// @param N
/// @param K
/// @param alpha
/// @param A
/// @param B
/// @param beta
/// @param C
/// @return
__global__ void sgemm_naive_row_major(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C);


/// @brief  Simplest GEMM kernel, each thread is responsible for an entry in the output matrix. Expect matrix data in
/// row major order
/// @param M
/// @param N
/// @param K
/// @param alpha
/// @param A
/// @param B
/// @param beta
/// @param C
/// @return
__global__ void sgemm_naive_column_major(int M, int N, int K, float alpha, const float *A,
    const float *B, float beta, float *C);