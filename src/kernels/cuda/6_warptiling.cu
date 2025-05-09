#include "kernels.hpp"
#include "types.hpp"
#include <stdio.h>
#include <assert.h>


#define cudaAssert(condition) \
  if (!(condition)) printf("Assertion %s failed!\n", #condition);


const int WARPSIZE = 32;

/// Test that two matrices held in shared memory are the same
template <const int M, const int N>
__device__ void test_matrix_equality_smem(float *A, float *B, Layout layout) {

  int threadsPerBlock = blockDim.x;

  if (layout == Layout::ColumnMajor) {
    for (int i = threadIdx.x; i < M * N; i += threadsPerBlock) {
      int row = i / N;
      int col = i % N;
      float a = A[col * M + row];
      float b = B[col * M + row];
      cudaAssert(fabsf(a - b) < 1e-4f);
    }
  } else if (layout == Layout::RowMajor) {
    for (int i = threadIdx.x; i < M * N; i += threadsPerBlock) {
      int row = i / N;
      int col = i % N;
      float a = A[row * N + col];
      float b = B[row * N + col];
      cudaAssert(fabsf(a - b) < 1e-4f);
    }
  }
}


/// This is known to work from 2D blocktiling code so can be used as a point of truth for testing
/// Shared memory loads
template <const int BM, const int BN, const int BK>
__device__ __forceinline__ void
load_from_gmem_cooperative(int M, int N, int K, const float *A, float *As, float *As_T,
                   const float *B, float *Bs, int bk) {

  int threadId = threadIdx.x;
  int threadsPerBlock = blockDim.x;

  for (int i = threadId; i < BM * BK; i += threadsPerBlock) {
    int row = i / BK;
    int col = i % BK;
    int globalRow = blockIdx.y * BM + row;
    int globalCol = bk + col;

    if (globalRow < M && globalCol < K)
      As[row * BK + col] = A[globalRow * K + globalCol];
    else
      As[row * BK + col] = 0.0f;
  }

  // Load tile of B into shared memory
  for (int i = threadId; i < BK * BN; i += threadsPerBlock) {
    int row = i / BN;
    int col = i % BN;
    int globalRow = bk + row;
    int globalCol = blockIdx.x * BN + col;

    if (globalRow < K && globalCol < N)
      Bs[row * BN + col] = B[globalRow * N + globalCol];
    else
      Bs[row * BN + col] = 0.0f;
  }

  // Now need to transpose As
  for (int i = threadId; i < BM * BK; i += threadsPerBlock) {

    // Assign threads to physical rows and columns again assuming row major
    // order of data i.e. adjacent data correspond to adjacent threads
    int row = i / BK;
    int col = i % BK;
    int globalRow =
        blockIdx.y * BM +
        row; // convert block index into units of threads, and add local row
    int globalCol = bk + col;

    if (globalRow < M && globalCol < K)
      As_T[col * BM + row] = As[row * BK + col];
  }

  __syncthreads();
}

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
    // As Reading in in row major order As[row * BK + col * 4]     = tmp.x;
    // As[row * BK + col * 4 + 1] = tmp.y;
    // As[row * BK + col * 4 + 2] = tmp.z;
    // As[row * BK + col * 4 + 3] = tmp.w;

    // However, to read in and transpose at the same time can switch into column
    // major order Read in and transpose at the same time
    As[(col * 4 + 0) * BM + row] = tmp.x;
    As[(col * 4 + 1) * BM + row] = tmp.y;
    As[(col * 4 + 2) * BM + row] = tmp.z;
    As[(col * 4 + 3) * BM + row] = tmp.w;
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


template <const int BM, const int BN, const int BK>
__device__ __forceinline__ void
test_load_from_gmem(int M, int N, int K, const float *A, const float *B) {

  const int threadId = threadIdx.x;
  const int threadsPerBlock = blockDim.x;

  __shared__ float As_expected[BM * BK];
  __shared__ float Bs_expected[BK * BN];
  __shared__ float As_expected_T[BK * BM];
  __shared__ float As_found_T[BM * BK];
  __shared__ float Bs_found[BK * BN];

  for (int bk = 0; bk < K; bk += BK) {
    load_from_gmem_cooperative<BM, BN, BK>(M, N, K, A, As_expected, As_expected_T, B, Bs_expected, bk);
    load_from_gmem_vectorised<BM, BN, BK>(M, N, K, A, As_found_T, B, Bs_found, bk);

    test_matrix_equality_smem<BM, BK>(As_found_T, As_expected_T, Layout::ColumnMajor);
    test_matrix_equality_smem<BK, BN>(Bs_expected, Bs_found, Layout::RowMajor);
  }


  // Test transposition

  // Test loading
}

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int WMITER, const int TM, const int TN>
__global__ void sgemm_warptiling(int M, int N, int K, float alpha, const float *A,
                                 const float *B, float beta, float *C) {

  __shared__ float As[BM * BK];
  __shared__ float tmp[BM * BK];
  __shared__ float Bs[BK * BN];

  // Inner loop is unrolled, as usual.
  // The two outer loops are over thread blocks
  const int threadId = threadIdx.x;

  // Calculate warp position within thread block
  const int warpId = threadId / WARPSIZE;
  const int warpRow = warpId / (BN / WN);
  const int warpCol = warpId % (BN / WN);

  // Calculate warp tile dimensions
  const int WSUBM = WM / WMITER; // Sizes of each subtile after split by WMITER/WNITER
  const int WSUBN = WN / WNITER; // These sizes are in units of physical entries

  // Thread position within warp subtile
  const int threadIdInWarp = threadId % WARPSIZE;
  const int threadRowInWarp = threadIdInWarp / (WSUBN / TN); // have to convert from units of physical entries to thread entries
  const int threadColInWarp = threadIdInWarp % (WSUBN / TN);

  // The warp subtiles are what would actually be handled by a WMMA call
  // Allocate registers for thread local data
  float regM[TM * WMITER] = {0.0};
  float regN[TN * WNITER] = {0.0f};
  float threadResults[TM*TN*WMITER*WNITER] = {0.0f};

  // Uncomment to test GMEM -> SMEM loading strategy
  // load_from_gmem_test<BM, BN, BK>(M, N, K, A, As, B, Bs, true);

  for (int bk = 0; bk < K; bk += BK) {
    // load_from_gmem_vectorised<BM, BN, BK>(M, N, K, A, As, B, Bs, bk);
    load_from_gmem_cooperative<BM, BN, BK>(M, N, K, A, tmp, As, B, Bs, bk);

    // Uncomment to benchmark old 2D blocktiling approach + manual transpose
    // load_from_gmem_old<BM, BN, BK>(M, N, K, A, As, As_T, B, Bs, bk);
    for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {

      // Loop over subtiles of warptile
      for (int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
        // For each subtile populate part of this thread's registers
        // with columns of As, remembering that A has been transposed therefore row/col swap positions in As
        for (int i = 0; i < TM; ++i) {
          regM[wSubRowIdx * TM + i] = As[(dotIdx * BM) + ((warpRow * WM) + (wSubRowIdx * WSUBM) + (threadRowInWarp * TM) + i)];
        }
      }

      for (int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
        // Populate with rows of Bs
        for (int i = 0; i < TN; ++i) {
          regN[wSubColIdx  * TN + i] = Bs[(dotIdx * BN) + ((warpCol * WN) + (wSubColIdx * WSUBN) + (threadColInWarp * TN) + i)];
        }
      }

      // Can now compute warp matrix multiplication in registers
      for (int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
        for (int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
          // Compute thread local results
          for (int i = 0; i < TM; ++i) {
            for (int j = 0; j < TN; ++j) {

              // This would be the case in 2D blocktiling
              // threadResults[i * TN + j] = regM[i] * regN[j];
              // Have to adjust for the number of results now each thread is responsible for
              // and therefore displace appropriately
              threadResults[((wSubRowIdx * TM + i) * (TN * WNITER)) + (wSubColIdx * TN + j)] +=
                regM[wSubRowIdx * TM + i] * regN[wSubColIdx * TN + j];
            }
          }
        }
      }
    }

    __syncthreads();
  }

  int cRow = blockIdx.y * BM;
  int cCol = blockIdx.x * BN;
  // Advance C to the appropriate Warp Row/COl
  C += (cRow + warpRow * WM) * N + cCol + warpCol * WN;

  // Only step left is to write the results from thread local results to global memory
  for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
      // Move pointer to right subtile of warp
      float *C_interim = C + (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN;

      // vector saving
      for (uint i = 0; i < TM; ++i) {
        for (uint j = 0; j < TN; j += 4) {

          // load C vector into registers
          float4 tmp = reinterpret_cast<float4 *>(
            &C_interim[(threadRowInWarp * TM + i) * N +
                       threadColInWarp * TN + j])[0];

          // perform GEMM update in reg
          const int linearIndexThreadResults = (wSubRowIdx * TM + i) * (WNITER * TN) +
                        wSubColIdx * TN + j;
          tmp.x = alpha * threadResults[linearIndexThreadResults + 0] + beta * tmp.x;
          tmp.y = alpha * threadResults[linearIndexThreadResults + 1] + beta * tmp.y;
          tmp.z = alpha * threadResults[linearIndexThreadResults + 2] + beta * tmp.z;
          tmp.w = alpha * threadResults[linearIndexThreadResults + 3] + beta * tmp.w;
          // write back
          reinterpret_cast<float4 *>(
              &C_interim[(threadRowInWarp * TM + i) * N +
                        threadColInWarp * TN + j])[0] = tmp;

        }
      }

      // for (uint i = 0; i < TM; ++i) {
      //   for (uint j = 0; j < TN; ++j) {

      //     const int threadResultsLinearIndex = (wSubRowIdx * TM + i) * (WNITER * TN) +
      //                   wSubColIdx * TN + j;

      //     float c_old = C_interim[(threadRowInWarp * TM + i) * N + (threadColInWarp * TN + j)];
      //     float c_new = alpha * threadResults[threadResultsLinearIndex] + beta * c_old;
      //     C_interim[(threadRowInWarp * TM + i) * N + (threadColInWarp * TN + j)] = c_new;
      //   }
      // }

      // Example of Scalar saving without advancing C to warp tile
        // for (uint j = 0; j < TN; ++j) {
        //   const int threadResultsLinearIndex = (wSubRowIdx * TM + i) * (WNITER * TN) +
        //                 wSubColIdx * TN + j;

        //   const int globalRow = blockIdx.y * BM +
        //                          warpRow * WM +
        //                          wSubRowIdx * WSUBM +
        //                          threadRowInWarp * TM +
        //                          i;

        //   const int globalCol = blockIdx.x * BN +
        //                          warpCol * WN +
        //                          wSubColIdx * WSUBN +
        //                          threadColInWarp * TN +
        //                          j;

        //   if (globalRow < M && globalCol < N) {
        //     const int globalLinearIndex = globalRow * N + globalCol;
        //     float c_old = C[globalLinearIndex];
        //     float c_new = alpha * threadResults[threadResultsLinearIndex] + beta * c_old;
        //     C[globalLinearIndex] = c_new;
        //   }
        // }
    }
  }
}

template __global__ void
sgemm_warptiling<64, 64, 64, 16, 16, 2, 2, 4, 4>(int, int, int, float, const float *,
                                              const float *, float, float *);

template __global__ void
sgemm_warptiling<64, 64, 32, 16, 16, 2, 2, 4, 4>(int, int, int, float, const float *,
                                              const float *, float, float *);

template __global__ void
sgemm_warptiling<64, 64, 32, 32, 32, 2, 2, 4, 4>(int, int, int, float, const float *,
                                              const float *, float, float *);

template __global__ void
sgemm_warptiling<64, 64, 64, 32, 32, 2, 2, 4, 4>(int, int, int, float, const float *,
                                              const float *, float, float *);

template __global__ void
sgemm_warptiling<64, 64, 64, 16, 16, 2, 1, 4, 4>(int, int, int, float, const float *,
                                              const float *, float, float *);

template __global__ void
sgemm_warptiling<64, 64, 32, 16, 16, 2, 1, 4, 4>(int, int, int, float, const float *,
                                              const float *, float, float *);

template __global__ void
sgemm_warptiling<64, 64, 32, 32, 32, 2, 1, 4, 4>(int, int, int, float, const float *,
                                              const float *, float, float *);

template __global__ void
sgemm_warptiling<64, 64, 64, 32, 32, 2, 1, 4, 4>(int, int, int, float, const float *,
                                              const float *, float, float *);

template __global__ void
sgemm_warptiling<128, 128, 16, 64, 64, 4, 1, 8, 4>(int, int, int, float, const float *,
                                              const float *, float, float *);