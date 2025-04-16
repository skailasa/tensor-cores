#include "kernels.hpp"


template <const int BLOCKSIZE>
__global__ void sgemm_smem_cache_blocking_row_major(int M, int N, int K, float alpha, const float *A,
        const float *B, float beta, float *C) {

    // The inner (block wise) row and column we're accessing in this thread
    const uint localCol = threadIdx.x;
    const uint localRow = threadIdx.y;

    // Row and column of the global output matrix C
    const uint globalCol = blockIdx.x * BLOCKSIZE + localCol;
    const uint globalRow = blockIdx.y * BLOCKSIZE + localRow;

    // Allocate buffer for current block in fast shared memory
    // Remember shared memory is per-block in CUDA
    __shared__ float As[BLOCKSIZE][BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE][BLOCKSIZE];

    // Loop along shared dimension, each thread block loads in a tile of BLOCKSIZE * BLOCKSIZE
    // from each of A and B, in a coalesced manner, into shared memory As and Bs resp.
    float tmp = 0.0;
    for (int bk = 0; bk < K; bk += BLOCKSIZE) {
        // Load A[row, bk + k] and B[bk + k, col] into shared memory

        float aVal = 0.0;
        float bVal = 0.0;
        if (globalRow < M && (bk + localCol) < K) {
            aVal = A[globalRow * K + (bk + localCol)]; // A[globalRow, bk+localCol]
        }

        if ((bk + localRow) < K && globalCol < N) {
            bVal = B[(bk + localRow) * N + globalCol]; // B[bk+localRow, globalCol]
        }

        As[localRow][localCol] = aVal;
        Bs[localRow][localCol] = bVal;

        // Loads should be coalesced as warp indices matches localCol (threadIdx.x)
        __syncthreads();

        // Compute dot product for the output value owned by this thread
        for (int k = 0; k < BLOCKSIZE; ++k) {
            tmp += As[localRow][k] * Bs[k][localCol];
        }

        __syncthreads();

    }

    if (globalRow < M && globalCol < N) {
        C[globalRow * N + globalCol] = alpha * tmp + beta * C[globalRow * N + globalCol];
    }

}


template <const int BLOCKSIZE>
__global__ void sgemm_smem_cache_blocking_column_major(int M, int N, int K, float alpha, const float *A,
        const float *B, float beta, float *C) {


    // The inner (block wise) row and column we're accessing in this thread
    const uint localCol = threadIdx.y;
    const uint localRow = threadIdx.x; // we now want the fastest running index to be for rows, for coalescing purposes

    // Row and column of the global output matrix C
    const uint globalCol = blockIdx.y * BLOCKSIZE + localCol;
    const uint globalRow = blockIdx.x * BLOCKSIZE + localRow;

    // Allocate buffer for current block in fast shared memory
    // Remember shared memory is per-block in CUDA
    __shared__ float As[BLOCKSIZE][BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE][BLOCKSIZE];


    float tmp = 0.0;
    for (int bk = 0; bk < K; bk += BLOCKSIZE) {

        float aVal = 0.0;
        float bVal = 0.0;
        if (globalRow < M && (bk + localCol) < K) {
            aVal = A[(bk + localCol) * M + globalRow]; // A[globalRow, bk + col]
        }

        if ((bk + localRow) < K && globalCol < N) {
            bVal = B[globalCol * K + (bk + localRow)]; // B[bk + row, globalCol]
        }

        As[localRow][localCol] = aVal;
        Bs[localRow][localCol] = bVal;

        // Loads should be coalesced as warp indices matches localCol (threadIdx.x)
        __syncthreads();

        // Compute dot product for the output value owned by this thread
        for (int k = 0; k < BLOCKSIZE; ++k) {
            tmp += As[localRow][k] * Bs[k][localCol];
        }

        __syncthreads();

    }
    if (globalRow < M && globalCol < N)
        C[globalCol * M + globalRow] = alpha * tmp + beta * C[globalCol * M + globalRow];

}

template __global__ void sgemm_smem_cache_blocking_row_major<32>(
    int, int, int, float, const float*, const float*, float, float*);

template __global__ void sgemm_smem_cache_blocking_column_major<32>(
    int, int, int, float, const float*, const float*, float, float*);