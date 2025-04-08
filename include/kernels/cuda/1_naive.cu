

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
                            const float *B, float beta, float *C) {

    // Thread indices within each block
    const uint col = blockDim.x * blockIdx.x + threadIdx.x;
    const uint row = blockDim.y * blockIdx.y + threadIdx.y;

    // Need to check if indices are within bounds of the matrix
    if (col < N && row < M) {
        float tmp = 0.0;

        for (int k = 0; k < K; k++) {
            tmp += A[row * K + k] * B[k * N + col];
        }

        C[row * N + col] = alpha * tmp + beta * C[row * N + col];
    }
}




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
    const float *B, float beta, float *C) {

    // Thread indices within each block
    const uint col = blockDim.x * blockIdx.x + threadIdx.x;
    const uint row = blockDim.y * blockIdx.y + threadIdx.y;

    // Need to check if indices are within bounds of the matrix
    if (col < N && row < M) {
        float tmp = 0.0;

        for (int k = 0; k < K; k++) {
            tmp += A[k * M + row] * B[col * K + k];
        }

        C[col * M + row] = alpha * tmp + beta * C[col * M + row];
    }
}

