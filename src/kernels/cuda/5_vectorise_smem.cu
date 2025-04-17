#include "kernels.hpp"

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemm_vectorise_smem_and_gmem_accesses_row_major(
    int M, int N, int K, float alpha, const float *A, const float *B,
    float beta, float *C) {

  // Add padding to avoid shared memory bank conflicts in A
  __shared__ float As[BK][BM + 1]; // +1 to break stride-32 conflict
  __shared__ float Bs[BK][BN];

  float regM[TM] = {0.0f};
  float regN[TN] = {0.0f};
  float threadResults[TM * TN] = {0.0f};

  const int threadId = threadIdx.x;
  const int threadsPerBlock = blockDim.x;

  const int threadRow = threadId / (BN / TN); // TM thread
  const int threadCol = threadId % (BN / TN); // TN thread

  // For vectorized load
  const int A_vecWidth = 4;
  const int B_vecWidth = 4;

  // How many float4s are needed to cover the tile
  const int numAFloat4s = (BK * BM) / A_vecWidth;
  const int numBFloat4s = (BK * BN) / B_vecWidth;

  for (int bk = 0; bk < K; bk += BK) {

    // === Load A with float4 and transpose ===
    for (int i = threadId; i < numAFloat4s; i += threadsPerBlock) {
      int flatIdx = i * A_vecWidth;
      int row = flatIdx / BK;
      int col = flatIdx % BK;

      int globalRow = blockIdx.y * BM + row;
      int globalCol = bk + col;

      float4 tmp = {0.f, 0.f, 0.f, 0.f};
      if (globalRow < M && globalCol + 3 < K) {
        tmp = *reinterpret_cast<const float4 *>(&A[globalRow * K + globalCol]);
      }

      // Write transposed into shared memory: As[col + offset][row]
      As[col + 0][row] = tmp.x;
      As[col + 1][row] = tmp.y;
      As[col + 2][row] = tmp.z;
      As[col + 3][row] = tmp.w;
    }

    // === Load B with float4 (no transpose) ===
    for (int i = threadId; i < numBFloat4s; i += threadsPerBlock) {
      int flatIdx = i * B_vecWidth;
      int row = flatIdx / BN;
      int col = flatIdx % BN;

      int globalRow = bk + row;
      int globalCol = blockIdx.x * BN + col;

      float4 tmp = {0.f, 0.f, 0.f, 0.f};
      if (globalRow < K && globalCol + 3 < N) {
        tmp = *reinterpret_cast<const float4 *>(&B[globalRow * N + globalCol]);
      }

      *reinterpret_cast<float4 *>(&Bs[row][col]) = tmp;
    }

    __syncthreads();

    // === Compute TM x TN tile ===
    for (int k = 0; k < BK; ++k) {
      for (int i = 0; i < TM; ++i)
        regM[i] = As[k][threadRow * TM + i]; // A is now As[dot][row]

      for (int i = 0; i < TN; ++i)
        regN[i] = Bs[k][threadCol * TN + i];

      for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j)
          threadResults[i * TN + j] += regM[i] * regN[j];
    }

    __syncthreads();
  }

  // === Write back ===
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
__global__ void sgemm_vectorise_smem_and_gmem_accesses_column_major(
    int M, int N, int K, float alpha, const float *A, const float *B,
    float beta, float *C) {}
