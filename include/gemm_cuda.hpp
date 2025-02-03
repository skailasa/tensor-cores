#include <cuda_runtime.h>
#include <cuda.h>
#include <mma.h>

using namespace nvcuda;

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

/// Naive kernel for BLAS3 operation
__global__ void dgemm_simple(int M, int N, int K, double alpha, const double *A, const double *B,
                             double beta, double *C) {

    // position in C that this thread is responsible for
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    // condition necesssary for non multiples of 32
    if (x < M && y < N) {
        double tmp = 0.;

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


/// @brief DGEMM using simple hierarchical tiling based kernel
template <
    unsigned int Bm,
    unsigned int Bn,
    unsigned int Bk,
    unsigned int Wm,
    unsigned int Wn,
    unsigned int Wk>
__global__ void dgemm_hierarchical_tiling(int M, int N, int K, double alpha, double* A, double* B,
                             double beta, double* C)
{
    constexpr unsigned int MMA_M_dim = 8;
    constexpr unsigned int MMA_N_dim = 8;
    constexpr unsigned int MMA_K_dim = 4;

    // calculate block/warp indices
    const unsigned int block_m = blockIdx.y;
    const unsigned int block_n = blockIdx.x;
    const unsigned int warp_m = threadIdx.y;
    const unsigned int warp_n = threadIdx.x / 32;

    const unsigned int A_stride = K;
    const unsigned int B_stride = N;
    const unsigned int CD_stride = N;

    // loop bounds, constexpr where possible allows for loop unrolling
    constexpr unsigned int mma_tiles_per_warp_k = Wk / MMA_K_dim;
    constexpr unsigned int mma_tiles_per_warp_m = Wm / MMA_M_dim;
    constexpr unsigned int mma_tiles_per_warp_n = Wn / MMA_N_dim;
    constexpr unsigned int warp_tiles_per_block_k = Bk / Wk;
    const unsigned int num_block_tiles_k = K/Bk;

    extern __shared__ double shmem[];
    double* A_block_smem = shmem;
    double* B_block_smem = &shmem[Bm * Bk];

    // Outer loop over block tiles
    for (unsigned int block_k = 0; block_k < num_block_tiles_k; block_k ++) {

        double* A_block_gmem = A + (block_m * Bm * A_stride) + (block_k * Bk);
        double* B_block_gmem = B + (block_k * Bk * B_stride) + (block_n * Bn);

        tileMemcpy(A_block_gmem, A_block_smem, K, Bm, Bk);
        tileMemcpy(B_block_gmem, B_block_smem, N, Bk, Bn);

        __syncthreads();

        // Inner loop over warp tiles
        for (unsigned int warp_k = 0; warp_k < warp_tiles_per_block_k; warp_k ++ ) {

            // Load from shared memory into register memory in preparation for compute phase
            // load block tiles into registers
            double* A_warp_tile = A_block_smem + (warp_m * Wm * Bk) + (warp_k * Wk);
            double* B_warp_tile = B_block_smem + (warp_k * Wk * Bn) + (warp_n * Wn);

            uint32_t A_warp_tile_byte_offset = cvta_to_shared_u32(A_warp_tile);
            uint32_t B_warp_tile_byte_offset = cvta_to_shared_u32(B_warp_tile);

            // pre-load tiles of A into registers
            for (unsigned int mma_m = 0; mma_m < mma_tiles_per_warp_m; mma_m ++) {
                for (unsigned int mma_k = 0; mma_k < mma_tiles_per_warp_k; mma_k +=) {
                    // byte offset to the top left of the mma tile
                    const unsigned int mma_tile_byte_offset = ((mma_m * MMA_M_dim * Bk) + (mma_k * MMA_K_dim)) * sizeof(double);

                    // byte offset to the start of this thread's slice of the mma tile
                    const unsigned int thread_byte_offset = (threadIdx.x % MMA_M_dim) * Bk * sizeof(double);

                    // calculate offset in bytes WRT to the start of our shared memory allocation
                    const unsigned int thread_offset_bytes = A_warp_tile_byte_offset + mma_tile_byte_offset + thread_byte_offset;



                }
            }

            // pre-load tiles of B into registers

        }
    }
}


// /// @brief DGEMM Using WMMA API, for Ada Generation RTX 6000.
// /// @param M
// /// @param N
// /// @param K
// /// @param alpha
// /// @param A
// /// @param B
// /// @param beta
// /// @param C
// /// @return
// __global__ void dgemm_wmma(const int M, int N, int K, double alpha, const double *A, const double *B,
//                                      double beta, double *C) {

//     // Tile using a 2D grid

//     // Determine the warp index
//     int warp_M = (blockIdx.x * blockDim.x + threadIdx.x) / 32; // Divide by warp size to find warp position
//     int warp_N = (blockIdx.y * blockDim.y + threadIdx.y);

//     // Create fragments
//     wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major> a_frag;
//     wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::row_major> b_frag;
//     wmma::fragment<wmma::accumulator, 8, 8, 4, double> c_frag;
//     wmma::fill_fragment(c_frag, 0.0f); // accumulate into 0

//     // Load data from shared memory into register file
//     wmma::load_matrix_sync(a_frag, A, K);
//     wmma::load_matrix_sync(b_frag, B, N);

//     // Perform matrix multiplication using tensor cores
//     wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

//     // Loop over shared dimension
//     for (uint32_t ki = 0; ki < K; ki += 4) {

//         // Find first element for mma matrices
//         uint32_t const matrix_mma_a_row_idx = warp_M * 8;
//         uint32_t const matrix_mma_a_col_idx = ki;

//         uint32_t const matrix_mma_b_row_idx = warp_N * 8;
//         uint32_t const matrix_mma_b_col_idx = ki;

//         uint32_t const matrix_mma_c_row_idx = warp_M * 8;
//         uint32_t const matrix_mma_c_col_idx = warp_N * 8;

//         // Check bounds
//         if (
//             matrix_mma_a_row_idx < M &&
//             matrix_mma_a_col_idx < K &&
//             matrix_mma_b_row_idx < K &&
//             matrix_mma_b_col_idx < N
//         )
//         {

//             // Create pointers to portion of global matrix to load into fragment
//             // We are assuming row major order
//             double const* A_p = A + matrix_mma_a_row_idx  * K + matrix_mma_a_col_idx;
//             double const* B_p = B + matrix_mma_b_row_idx  * N + matrix_mma_b_col_idx;

//             // Load into registers
//             wmma::load_matrix_sync(a_frag, A_p, K);
//             wmma::load_matrix_sync(b_frag, B_p, N);

//             // Perform matrix mult
//             wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

//             double const* C_p = C + matrix_mma_c_row_idx * N + matrix_mma_c_col_idx;

//             wmma::store_matrix_sync(C_p, c_frag, N, wmma::mem_row_major);

//         }

//     }
// }