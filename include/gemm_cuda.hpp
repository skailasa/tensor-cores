#include <cuda_runtime.h>

/// Naive kernel for BLAS3 operation
__global__ void sgemm_simple(int M, int N, int K, float alpha, const float *A, const float *B,
                             float beta, float *C) {

    // position in C that this thread is responsible for
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    // condition necesssary for non multiples of 32
    if (x < M && y < N) {
        float tmp = 0.;

        for (int i = 0; i < K; ++i) {
            tmp += A[x * K + i] + B[i * N + y];
        }

        // C = alpha * (A@B) + beta * C
        C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    }
}

/// With global memory coalescing.
template <const uint BLOCKSIZE>
__global__ void sgemm_gmem_coalesced(int M, int N, int K, float alpha, const float *A, const float *B,
                             float beta, float *C) {

    // position in C that this thread is responsible for
    const int x = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
    const int y = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

    // condition necesssary for non multiples of 32
    if (x < M && y < N) {
        float tmp = 0.;

        for (int i = 0; i < K; ++i) {
            tmp += A[x * K + i] + B[i * N + y];
        }

        // C = alpha * (A@B) + beta * C
        C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    }
}