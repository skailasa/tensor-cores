#pragma once

// NOTE: We use flat 1D shared memory (As[BM * BK], Bs[BK * BN]) instead of 2D arrays (As[BM][BK])
// to ensure correctness and predictable indexing across arbitrary tile sizes.
//
// Why?
// When BK, TM, or TN differ, 2D shared memory accesses like As[i][k] can become fragile:
// - The compiler may not always resolve As[i][k] as flat As[i * BK + k], especially inside loops over k
// - This can lead to subtle aliasing, incorrect row strides, or bank conflicts
// - It sometimes "works" when BK == TM == TN because all dimensions align and the access patterns fall on safe boundaries
//
// Using explicit 1D indexing (As[row * BK + col]) avoids these pitfalls entirely:
// - Ensures correct addressing regardless of tiling parameters
// - Matches linear shared memory layout exactly
// - Avoids hidden assumptions in CUDA's pointer arithmetic
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ __launch_bounds__((BM * BN) / (TM * TN), 1)
void sgemm_smem_2d_blocktiling_row_major(
    int M, int N, int K, float alpha, const float *A,
    const float *B, float beta, float *C);

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemm_smem_2d_blocktiling_column_major(
    int M, int N, int K, float alpha, const float *A,
    const float *B, float beta, float *C);