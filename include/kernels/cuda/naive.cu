

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
__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {

    // Thread indices within each block
    const uint x = blockDim.x * blockIdx.x + threadIdx.x;
    const uint y = blockDim.y * blockIdx.y + threadIdx.y;

    // Need to check if indices are within bounds of the matrix
    if (x < N && y < M) {
        float tmp = 0.0;

        for (int i = 0; i < K; i++) {
            tmp += A[y * K + i] * B[i * N + x];
        }

        C[y * N + x] = alpha * tmp + beta * C[y * N + x];
    }
}

