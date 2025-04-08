

template <const int BLOCKSIZE>
__global__ void sgemm_smem_cache_blocking_row_major(int M, int N, int K, float alpha, const float *A,
                                      const float *B, float beta, float *C) {

    // The output block this thread block is responsible for (of C)
    const uint cColBlock = blockIdx.x;
    const uint cRowBlock = blockIdx.y;

    // The inner (block wise) row and column we're accessing in this thread
    const uint threadCol = threadIdx.x;
    const uint threadRow = threadIdx.y;

    // Row and column of the global output matrix C
    const uint row = cRowBlock * BLOCKSIZE + threadRow;
    const uint col = cColBlock * BLOCKSIZE + threadCol;

    // Allocate buffer for current block in fast shared memory
    // Remember shared memory is per-block in CUDA
    __shared__ float As[BLOCKSIZE][BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE][BLOCKSIZE];

    // Loop along shared dimension, each thread block loads in a tile of BLOCKSIZE * BLOCKSIZE
    float tmp = 0.0;
    for (int bk = 0; bk < K; bk += BLOCKSIZE) {
        // Load A[row, bk + k] and B[bk + k, col] into shared memory

        float aVal = 0.0;
        float bVal = 0.0;
        if (row < M && (bk + threadCol) < K) {
            aVal = A[row * K + (bk + threadCol)];
        }

        if ((bk + threadRow) < K && col < N) {
            bVal = B[(bk + threadRow) * N + col];
        }

        As[threadRow][threadCol] = aVal;
        Bs[threadRow][threadCol] = bVal;

        // Loads should be coalesced as warp indices matches threadCol (threadIdx.x)
        __syncthreads();

        // Compute dot product for the output value owned by this thread
        for (int k = 0; k < BLOCKSIZE; ++k) {
            tmp += As[threadRow][k] * Bs[k][threadCol];
        }

        __syncthreads();

    }

    if (row < M && col < N) {
        C[row * N + col] = alpha * tmp + beta * C[row * N + col];
    }

}


template <const int BLOCKSIZE>
__global__ void sgemm_smem_cache_blocking_column_major(int M, int N, int K, float alpha, const float *A,
                                      const float *B, float beta, float *C) {


}