#include <cuda_runtime.h>


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