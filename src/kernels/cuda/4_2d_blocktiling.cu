#include "kernels.hpp"

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
    const float *B, float beta, float *C) {

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    float regM[TM] = {0.0f};
    float regN[TN] = {0.0f};
    float threadResults[TM * TN] = {0.0f};

    const int threadId = threadIdx.x;
    const int threadsPerBlock = blockDim.x;

    const int threadRow = threadId / (BN / TN);
    const int threadCol = threadId % (BN / TN);

    for (int bk = 0; bk < K; bk += BK) {
        // Load tile of A into shared memory
        for (int i = threadId; i < BM * BK; i += threadsPerBlock) {
            int row = i / BK;
            int col = i % BK;
            int globalRow = blockIdx.y * BM + row;
            int globalCol = bk + col;

            if (globalRow < M && globalCol < K)
                As[row * BK + col] = A[globalRow * K + globalCol];
            else
                As[row * BK + col] = 0.0f;
        }

        // Load tile of B into shared memory
        for (int i = threadId; i < BK * BN; i += threadsPerBlock) {
            int row = i / BN;
            int col = i % BN;
            int globalRow = bk + row;
            int globalCol = blockIdx.x * BN + col;

            if (globalRow < K && globalCol < N)
                Bs[row * BN + col] = B[globalRow * N + globalCol];
            else
                Bs[row * BN + col] = 0.0f;
        }

        __syncthreads();

        // Compute TM x TN tile
        for (int k = 0; k < BK; ++k) {
            for (int i = 0; i < TM; ++i)
                regM[i] = As[(threadRow * TM + i) * BK + k];

            for (int i = 0; i < TN; ++i)
                regN[i] = Bs[k * BN + threadCol * TN + i];

            for (int i = 0; i < TM; ++i)
                for (int j = 0; j < TN; ++j)
                    threadResults[i * TN + j] += regM[i] * regN[j];
        }

        __syncthreads();
    }

    // Write back the result to global memory
    for (int i = 0; i < TM; ++i) {
        int globalRow = blockIdx.y * BM + threadRow * TM + i;
        if (globalRow >= M) continue;

        for (int j = 0; j < TN; ++j) {
            int globalCol = blockIdx.x * BN + threadCol * TN + j;
            if (globalCol >= N) continue;

            int idx = globalRow * N + globalCol;
            C[idx] = alpha * threadResults[i * TN + j] + beta * C[idx];
        }
    }
}


template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemm_smem_2d_blocktiling_column_major(
    int M, int N, int K, float alpha, const float *A,
    const float *B, float beta, float *C) {
}


template __global__ void sgemm_smem_2d_blocktiling_row_major<64, 64, 16, 4, 4>(int, int, int, float, const float*, const float*, float, float*);
template __global__ void sgemm_smem_2d_blocktiling_row_major<64, 64, 32, 4, 4>(int, int, int, float, const float*, const float*, float, float*);
template __global__ void sgemm_smem_2d_blocktiling_row_major<64, 64, 64, 4, 4>(int, int, int, float, const float*, const float*, float, float*);
template __global__ void sgemm_smem_2d_blocktiling_column_major<64, 64, 16, 4, 4>(int, int, int, float, const float*, const float*, float, float*);