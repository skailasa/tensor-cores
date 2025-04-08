
/// Still each thread is responsible for a single entry of the output matrix, but now consecutive threads
/// read in the same row of A (which is broadcast across a warp) and subsequent columns of B, ensuring
/// global memory read coalescing

template <const uint BLOCKSIZE>
__global__ void sgemm_gmem_coalescing(int M, int N, int K, float alpha, const float *A,
                                      const float *B, float beta, float *C) {

    const uint x = BLOCKSIZE * blockIdx.x + (threadIdx.x / BLOCKSIZE);
    const uint y = BLOCKSIZE * blockIdx.y + (threadIdx.x % BLOCKSIZE);

  // if statement is necessary to make things work under tile quantization
  if (x < M && y < N) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      tmp += A[y * K + i] * B[i * N + x];
    }

    C[y * N + x] = alpha * tmp + beta * C[y * N + x];
  }
}