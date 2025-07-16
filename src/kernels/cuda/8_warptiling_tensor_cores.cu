#include "kernels.hpp"
#include "mma.h"
#include "types.hpp"

const int WARPSIZE = 32;

using namespace nvcuda;

/// A vectorised loading strategy where each thread loads 32*4 = 128 bit = 16
/// Bytes per-shared memory access
template <const int BM, const int BN, const int BK>
__device__ __forceinline__ void
load_from_gmem_vectorised(int M, int N, int K, half *A, half *As, half *B,
                          half *Bs, int bk) {

  const int threadId = threadIdx.x;
  const int threadsPerBlock = blockDim.x;

  // Load A into As, assume row major order
  for (int i = threadId; i < ((BM * BK) / 8); i += threadsPerBlock) {
    // Basically indexing by saying that number of cols 4x smaller as we read in
    // 4 floats at a time of row major data. The number of rows is still the
    // same, and therefore so is the code to calculate the physical row index
    // For future
    int row = i / (BK / 8);
    int col = i % (BK / 8);
    int globalRow =
        blockIdx.y * BM +
        row; // convert block index into units of threads, and add local row
    int globalCol = bk + col * 8;

    float4 tmp = {0.0f, 0.0f, 0.0f, 0.0f};

    if (globalRow < M && globalCol + 7 < K) {
      tmp = reinterpret_cast<float4 *>(&A[globalRow * K + globalCol])[0];
    }

    half *tmp_half = reinterpret_cast<half *>(&tmp);

    As[(col * 8 + 0) * BM + row] = tmp_half[0];
    As[(col * 8 + 1) * BM + row] = tmp_half[1];
    As[(col * 8 + 2) * BM + row] = tmp_half[2];
    As[(col * 8 + 3) * BM + row] = tmp_half[3];
    As[(col * 8 + 4) * BM + row] = tmp_half[4];
    As[(col * 8 + 5) * BM + row] = tmp_half[5];
    As[(col * 8 + 6) * BM + row] = tmp_half[6];
    As[(col * 8 + 7) * BM + row] = tmp_half[7];
  }

  // // Load B into Bs
  for (int i = threadId; i < ((BK * BN) / 8); i += threadsPerBlock) {
    int row = i / (BN / 8);
    int col = i % (BN / 8);
    int globalRow = bk + row;
    int globalCol = blockIdx.x * BN + col * 8;

    float4 tmp = {0.0f, 0.0f, 0.0f, 0.0f};

    if (globalRow < K && globalCol + 7 < N) {
      tmp = reinterpret_cast<float4 *>(&B[globalRow * N + globalCol])[0];
    }

    half *tmp_half = reinterpret_cast<half *>(&tmp);

    Bs[row * BN + col * 8] = tmp_half[0];
    Bs[row * BN + col * 8 + 1] = tmp_half[1];
    Bs[row * BN + col * 8 + 2] = tmp_half[2];
    Bs[row * BN + col * 8 + 3] = tmp_half[3];
    Bs[row * BN + col * 8 + 4] = tmp_half[4];
    Bs[row * BN + col * 8 + 5] = tmp_half[5];
    Bs[row * BN + col * 8 + 6] = tmp_half[6];
    Bs[row * BN + col * 8 + 7] = tmp_half[7];
  }

  __syncthreads();
}

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WK, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    hgemm_warptiling_tensor_cores(int M, int N, int K, half alpha, half *A,
                                  half *B, half beta, half *C)

{
  constexpr unsigned int MMA_M = 16;
  constexpr unsigned int MMA_N = 16;
  constexpr unsigned int MMA_K = 16;

  constexpr unsigned int mma_tiles_per_warp_k = WK / MMA_K;
  constexpr unsigned int mma_tiles_per_warp_m = WM / MMA_M;
  constexpr unsigned int mma_tiles_per_warp_n = WN / MMA_N;

  constexpr unsigned int warp_tiles_per_block_k = BK / WK;
  const unsigned int num_block_tiles_k = K / BK;

  // Calculate block and warp indices
  const int threadId = threadIdx.x;
  const int warpId = threadId / WARPSIZE;
  const int warpRow = warpId / (BN / WN);
  const int warpCol = warpId % (BN / WN);

  // Allocate shared memory to load in A and B
  __shared__ half As[BM * BK];
  __shared__ half Bs[BK * BN];

  // Allocate registers for temporary loads
  wmma::fragment<wmma::matrix_a, MMA_M, MMA_N, MMA_K, __half, wmma::col_major> A_reg[mma_tiles_per_warp_m][mma_tiles_per_warp_k];
  wmma::fragment<wmma::matrix_b, MMA_M, MMA_N, MMA_K, __half, wmma::row_major> B_reg[mma_tiles_per_warp_k][mma_tiles_per_warp_n];

  // Allocate register in which to accumulate results
  wmma::fragment<wmma::accumulator, MMA_M, MMA_N, MMA_K, half> C_reg[mma_tiles_per_warp_m][mma_tiles_per_warp_n];

  // Set accumulator to zero
  for (int i = 0; i < mma_tiles_per_warp_m; ++i) {
    for (int j = 0; j < mma_tiles_per_warp_n; ++j) {
      wmma::fill_fragment(C_reg[i][j], 0.0);
    }
  }

  // Now perform k-loop over blocks (outermost loop)
  for (int bk = 0; bk < K; bk += BK) {

    // Load current block into shared memory using vectorised loads
    load_from_gmem_vectorised<BM, BN, BK>(M, N, K, A, As, B, Bs, bk);

    // middle loop over warp tiles
    for (int warp_k = 0; warp_k < warp_tiles_per_block_k; ++warp_k) {

      // Need to load operands into warp local registers

      // Find displacements (As is column major)
      half* A_warp_tile = &As[(warpRow * WM) + (warp_k * WK) * BM];
      half* B_warp_tile = &Bs[(warp_k * WK)*BN + (warpCol * WN)];

      for (int i = 0; i < mma_tiles_per_warp_m; ++i) {
        for (int j = 0; j < mma_tiles_per_warp_k; ++j) {

          // Find pointer to MMA tile from shared memory
          int mma_tile_offset = (i * MMA_M) + (j * MMA_K) * BM;
          half* A_mma_tile = A_warp_tile + mma_tile_offset;
          // Load into register
          wmma::load_matrix_sync(A_reg[i][j], A_mma_tile, BM);
        }
      }

      for (int i = 0; i < mma_tiles_per_warp_k; ++i) {
        for (int j = 0; j < mma_tiles_per_warp_n; ++j) {

          // Find pointer to MMA tile from shared memory
          int mma_tile_offset = (i * MMA_M)*BN + (j * MMA_K);

          half* B_mma_tile = B_warp_tile + mma_tile_offset;
          // Load into register
          wmma::load_matrix_sync(B_reg[i][j], B_mma_tile, BN);
        }
      }

      // Perform matrix multiplication in registers as an outer product
      for (int k = 0; k < mma_tiles_per_warp_k; ++k) {
        for (int j = 0; j < mma_tiles_per_warp_n; ++j) {
          for (int i = 0; i < mma_tiles_per_warp_m; ++i ) {

            wmma::mma_sync(C_reg[i][j], A_reg[i][k], B_reg[k][i], C_reg[i][j]);

          }
        }
      }
    }
    __syncthreads();
  }

  // Save to global memory
  // Calculate pointers to this warps's output tiles

  // Find block tile (global memory pointer)
  half* C_block_tile = &C[(blockIdx.y * BM)*N + (blockIdx.x*BN)]; // corresponding output block of C

  // Find corresponding warp tile (gmem)
  half* C_warp_tile = C_block_tile + (warpRow * WM * N) + (warpCol * WN);

  // Apply scaling
  for (int i = 0; i < mma_tiles_per_warp_m; ++i) {
    for (int j = 0; j < mma_tiles_per_warp_n; ++j) {

      half* C_mma_tile = C_warp_tile + (i * MMA_M * N) + (j * MMA_N);

      wmma::fragment<wmma::accumulator, MMA_M, MMA_N, MMA_K, half> C_orig;
      wmma::load_matrix_sync(C_orig, C_mma_tile, N, wmma::mem_row_major);

      // Something here for scaling
      for (int t = 0; t < C_reg[i][j].num_elements; ++t) {
        C_reg[i][j].x[t] = alpha * C_reg[i][j].x[t] + beta * C_orig.x[t];
      }

      // Something to store in global memory
      wmma::store_matrix_sync(C_mma_tile, C_reg[i][j], N, wmma::mem_row_major);
    }
  }
}



template __global__ void
hgemm_warptiling_tensor_cores<64, 64, 64, 16, 16, 16, 128>(
    int, int, int, half, half *, half *, half, half *);

template __global__ void
hgemm_warptiling_tensor_cores<64, 64, 32, 32, 32, 32, 128>(
    int, int, int, half, half *, half *, half, half *);
