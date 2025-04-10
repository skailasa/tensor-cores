template <const int BM, const int BN, const int BK, const int TM>
__global__ void sgemm_smem_1d_blocktiling_row_major(
    int M, int N, int K, float alpha, const float *A,
    const float *B, float beta, float *C) {

    // Shared memory tiles for A and B
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    // Each thread computes TM values in the output
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

}