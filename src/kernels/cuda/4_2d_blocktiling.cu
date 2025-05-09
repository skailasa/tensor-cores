#include "kernels.hpp"
/// A vectorised loading strategy where each thread loads 32*4 = 128 bit = 16 Bytes per-shared memory access
template <const int BM, const int BN, const int BK>
__device__ __forceinline__ void load_from_gmem_vectorised(int M, int N, int K, const float *A,
                                               float *As, const float *B, float *Bs,
                                               int bk) {

  const int threadId = threadIdx.x;
  const int threadsPerBlock = blockDim.x;

  // Load A into As, assume row major order
  for (int i = threadId; i < ((BM * BK) / 4); i += threadsPerBlock) {
    // Basically indexing by saying that number of cols 4x smaller as we read in
    // 4 floats at a time of row major data. The number of rows is still the
    // same, and therefore so is the code to calculate the physical row index
    // For future
    int row = i / (BK / 4);
    int col = i % (BK / 4);
    int globalRow =
        blockIdx.y * BM +
        row; // convert block index into units of threads, and add local row
    int globalCol = bk + col * 4;

    float4 tmp = {0.0f, 0.0f, 0.0f, 0.0f};

    if (globalRow < M && globalCol + 3 < K) {
      tmp = reinterpret_cast<const float4 *>(&A[globalRow * K + globalCol])[0];
    }

    // N.B For future me, this commented out code basically reads into rows of
    // As Reading in in row major order
    As[row * BK + col * 4]     = tmp.x;
    As[row * BK + col * 4 + 1] = tmp.y;
    As[row * BK + col * 4 + 2] = tmp.z;
    As[row * BK + col * 4 + 3] = tmp.w;

    // However, to read in and transpose at the same time can switch into column
    // major order Read in and transpose at the same time
    // As[(col * 4 + 0) * BM + row] = tmp.x;
    // As[(col * 4 + 1) * BM + row] = tmp.y;
    // As[(col * 4 + 2) * BM + row] = tmp.z;
    // As[(col * 4 + 3) * BM + row] = tmp.w;
  }

  // // Load B into Bs
  for (int i = threadId; i < ((BK * BN) / 4); i += threadsPerBlock) {
    int row = i / (BN / 4);
    int col = i % (BN / 4);
    int globalRow = bk + row;
    int globalCol = blockIdx.x * BN + col * 4;

    float4 tmp = {0.0f, 0.0f, 0.0f, 0.0f};

    if (globalRow < K && globalCol + 3 < N) {
      tmp = reinterpret_cast<const float4 *>(&B[globalRow * N + globalCol])[0];
    }

    Bs[row * BN + col * 4] = tmp.x;
    Bs[row * BN + col * 4 + 1] = tmp.y;
    Bs[row * BN + col * 4 + 2] = tmp.z;
    Bs[row * BN + col * 4 + 3] = tmp.w;
  }

  __syncthreads();
}

// NOTE: We use flat 1D shared memory (As[BM * BK], Bs[BK * BN]) instead of 2D
// arrays (As[BM][BK]) to ensure correctness and predictable indexing across
// arbitrary tile sizes.
//
// Why?
// When BK, TM, or TN differ, 2D shared memory accesses like As[i][k] can become
// fragile:
// - The compiler may not always resolve As[i][k] as flat As[i * BK + k],
// especially inside loops over k
// - This can lead to subtle aliasing, incorrect row strides, or bank conflicts
// - It sometimes "works" when BK == TM == TN because all dimensions align and
// the access patterns fall on safe boundaries
//
// Using explicit 1D indexing (As[row * BK + col]) avoids these pitfalls
// entirely:
// - Ensures correct addressing regardless of tiling parameters
// - Matches linear shared memory layout exactly
// - Avoids hidden assumptions in CUDA's pointer arithmetic
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ __launch_bounds__(
    (BM * BN) / (TM * TN),
    1) void sgemm_smem_2d_blocktiling_row_major(int M, int N, int K,
                                                float alpha, const float *A,
                                                const float *B, float beta,
                                                float *C) {

  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  float regM[TM] = {0.0f};
  float regN[TN] = {0.0f};
  float threadResults[TM * TN] = {0.0f};

  const int threadId = threadIdx.x;

  const int threadRow = threadId / (BN / TN);
  const int threadCol = threadId % (BN / TN);

  for (int bk = 0; bk < K; bk += BK) {

    load_from_gmem_vectorised<BM, BN, BK>(M, N, K, A, As, B, Bs, bk);
    // // Load tile of A into shared memory
    // for (int i = threadId; i < BM * BK; i += threadsPerBlock) {
    //   int row = i / BK;
    //   int col = i % BK;
    //   int globalRow = blockIdx.y * BM + row;
    //   int globalCol = bk + col;

    //   if (globalRow < M && globalCol < K)
    //     As[row * BK + col] = A[globalRow * K + globalCol];
    //   else
    //     As[row * BK + col] = 0.0f;
    // }

    // // Load tile of B into shared memory
    // for (int i = threadId; i < BK * BN; i += threadsPerBlock) {
    //   int row = i / BN;
    //   int col = i % BN;
    //   int globalRow = bk + row;
    //   int globalCol = blockIdx.x * BN + col;

    //   if (globalRow < K && globalCol < N)
    //     Bs[row * BN + col] = B[globalRow * N + globalCol];
    //   else
    //     Bs[row * BN + col] = 0.0f;
    // }

    // __syncthreads();

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
    if (globalRow >= M)
      continue;

    for (int j = 0; j < TN; ++j) {
      int globalCol = blockIdx.x * BN + threadCol * TN + j;
      if (globalCol >= N)
        continue;

      int idx = globalRow * N + globalCol;
      C[idx] = alpha * threadResults[i * TN + j] + beta * C[idx];
    }
  }
}

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void
sgemm_smem_2d_blocktiling_column_major(int M, int N, int K, float alpha,
                                       const float *A, const float *B,
                                       float beta, float *C) {}

template __global__ void sgemm_smem_2d_blocktiling_row_major<64, 64, 16, 4, 4>(
    int, int, int, float, const float *, const float *, float, float *);
template __global__ void sgemm_smem_2d_blocktiling_row_major<64, 64, 32, 4, 4>(
    int, int, int, float, const float *, const float *, float, float *);
template __global__ void sgemm_smem_2d_blocktiling_row_major<64, 64, 64, 4, 4>(
    int, int, int, float, const float *, const float *, float, float *);
template __global__ void
sgemm_smem_2d_blocktiling_column_major<64, 64, 16, 4, 4>(int, int, int, float,
                                                         const float *,
                                                         const float *, float,
                                                         float *);