#include "kernels.hpp"

template <const int BM, const int BN, const int BK, const int TM>
__global__ void sgemm_smem_1d_blocktiling_row_major(
    int M, int N, int K, float alpha, const float *A,
    const float *B, float beta, float *C) {

    // Shared memory tiles for A and B
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    // Each thread computes TM values in the output
    // This is stored in registers
    float threadResults[TM] = {0.0f};

    int threadId = threadIdx.x;
    int threadsPerBlock = blockDim.x;

    int threadRow = threadIdx.x / BN;
    int threadCol = threadIdx.x % BN;

    // Need to still load As and Bs tiles into shared memory by looping along shared dimension
    for (int bk = 0; bk < K; bk += BK) {

        // Load As

        // Only threads with tid UP TO BM*BK will participate, so at worst one thread will be responsible for
        // loading one element
        for (int i = threadId; i < BM * BK; i += threadsPerBlock) {
            int row = i / BK; // rows are of length BK
            int col = i % BK; // position within each row
            int globalRow = blockIdx.y * BM + row;
            int globalCol = bk + col;

            if (globalRow < M && globalCol < K) {
                As[row][col] = A[globalRow*K + globalCol];
            }
            else {
                As[row][col] = 0.0;
            }
        }

        // Load Bs
        for (int i = threadId; i < BK * BN; i += threadsPerBlock) {
            int row = i / BN;
            int col = i % BN;
            int globalRow = bk+row;
            int globalCol = blockIdx.x * BN + col;

            if (globalRow < K && globalCol < N) {
                Bs[row][col] = B[globalRow*N + globalCol];
            }
            else {
                Bs[row][col] = 0.0;
            }
        }

        __syncthreads();

        // Calculate results per thread
        // Strategy, fix column of B, outer loop over rows of B
        // Inner loop over each corresponding col value of A in each row
        // i.e. loop over rows of As, same index as result index. Coalesced loads from B
        // Non coalesced loads from As
        for (int k = 0; k < BK; ++k) {
            float tmpB = Bs[k][threadCol];
            for (int i = 0; i < TM; ++i) {
                int row = threadRow * TM + i;
                threadResults[i] += As[row][k] * tmpB;
            }
        }

        __syncthreads();
    }

    // write results
    for (int resIdx = 0; resIdx < TM; ++resIdx) {
        int globalRow = blockIdx.y * BM + threadRow * TM + resIdx;
        int globalCol = blockIdx.x * BN + threadCol;

        C[globalRow * N + globalCol] = alpha * threadResults[resIdx] + beta * C[globalRow * N + threadCol];
    }

}


template <const int BM, const int BN, const int BK, const int TM>
__global__ void sgemm_smem_1d_blocktiling_column_major(int M, int N, int K, float alpha, const float *A,
    const float *B, float beta, float *C) {

        __shared__ float As[BK][BM]; // Now in column major order
        __shared__ float Bs[BN][BK];

        // This threads result, still responsible for column of TM of output matrix C per thread
        float threadResults[TM] = {0.0f};

        // Now running this over columns of each output matrix, rather than rows like above
        int threadId = threadIdx.x;
        int threadsPerBlock = blockDim.x;

        int threadRow = threadIdx.x % (BM/TM);
        int threadCol = threadIdx.x / (BM/TM);

        // Still need to load As and Bs into shared memory, looping along shared
        // Loop over shared dimension still in tile size
        for (int bk = 0; bk < K; bk += BK) {

            // Load As
            for (int i = threadId; i < BM * BK; i += threadsPerBlock) {
                int row = i % BM;
                int col = i / BM;
                int globalRow = blockIdx.y * BM + row;
                int globalCol = bk + col;

                if (globalRow < M && globalCol < K) {
                    As[col][row] = A[globalCol*M + globalRow];
                } else {
                    As[col][row] = 0.0f;
                }
            }

            // Load Bs
            for (int i = threadId; i < BK * BN; i += threadsPerBlock) {
                int row = i % BK;
                int col = i / BK;
                int globalRow = bk + row;
                int globalCol = blockIdx.x * BN + col;

                if (globalRow < K && globalCol < N) {
                    Bs[col][row] = B[globalCol * K + globalRow];
                } else {
                    Bs[col][row] = 0.0f;
                }
            }

            __syncthreads();

            // Calculate per-thread result
            for (int k = 0; k < BK; ++k) {
                float tmpB = Bs[threadCol][k];
                for (int i = 0; i < TM; ++i) {
                    int row = threadRow * TM + i;
                    threadResults[i] += As[k][row] * tmpB;
                }
            }

            __syncthreads();

        }

        // Write results
        for (int resIdx = 0; resIdx < TM; ++resIdx) {
            int globalRow = blockIdx.y * BM + threadRow * TM + resIdx;
            int globalCol = blockIdx.x * BN + threadCol;

            C[globalRow + globalCol * M] = alpha * threadResults[resIdx] + beta *  C[globalRow + globalCol * M];
        }
}

template __global__ void sgemm_smem_1d_blocktiling_row_major<64, 64, 16, 4>(int, int, int, float, const float*, const float*, float, float*);
template __global__ void sgemm_smem_1d_blocktiling_column_major<64, 64, 16, 4>(int, int, int, float, const float*, const float*, float, float*);
