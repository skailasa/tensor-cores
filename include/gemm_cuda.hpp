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


// /// @brief DGEMM using simple hierarchical tiling based kernel
// template <
//     unsigned int Bm,
//     unsigned int Bn,
//     unsigned int Bk,
//     unsigned int Wm,
//     unsigned int Wn,
//     unsigned int Wk>
// __global__ void dgemm_wmma(int M, int N, int K, double alpha, double* A, double* B,
//                              double beta, double* C)
// {

// }


/// @brief DGEMM Using WMMA API, for Ada Generation RTX 6000.
/// @param M
/// @param N
/// @param K
/// @param alpha
/// @param A
/// @param B
/// @param beta
/// @param C
/// @return
__global__ void dgemm_wmma(const int M, int N, int K, double alpha, const double *A, const double *B,
                                     double beta, double *C) {

    // Tile using a 2D grid

    // Determine the warp index
    int warp_M = (blockIdx.x * blockDim.x + threadIdx.x) / 32; // Divide by warp size to find warp position
    int warp_N = (blockIdx.y * blockDim.y + threadIdx.y);

    // Create fragments
    wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 8, 8, 4, double> c_frag;
    wmma::fill_fragment(c_frag, 0.0f); // accumulate into 0


    // Loop over shared dimension
    for (uint32_t ki = 0; ki < K; ki += 4) {

        // Find first element for mma matrices
        uint32_t const matrix_mma_a_row_idx = warp_M * 8;
        uint32_t const matrix_mma_a_col_idx = ki;

        uint32_t const matrix_mma_b_row_idx = warp_N * 8;
        uint32_t const matrix_mma_b_col_idx = ki;

        uint32_t const matrix_mma_c_row_idx = warp_M * 8;
        uint32_t const matrix_mma_c_col_idx = warp_N * 8;

        // Check bounds
        if (
            matrix_mma_a_row_idx < M &&
            matrix_mma_a_col_idx < K &&
            matrix_mma_b_row_idx < K &&
            matrix_mma_b_col_idx < N
        )
        {

            // Create pointers to portion of global matrix to load into fragment
            // We are assuming row major order
            double const* A_p = A + matrix_mma_a_row_idx  * K + matrix_mma_a_col_idx;
            double const* B_p = B + matrix_mma_b_row_idx  * N + matrix_mma_b_col_idx;

            // Load into registers
            wmma::load_matrix_sync(a_frag, A_p, K);
            wmma::load_matrix_sync(b_frag, B_p, N);

            // Perform matrix mult
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

            double* C_p = C + matrix_mma_c_row_idx * N + matrix_mma_c_col_idx;

            wmma::store_matrix_sync(C_p, c_frag, N, wmma::mem_row_major);

        }

        if (matrix_mma_c_row_idx < M && matrix_mma_b_col_idx < N) {

            double const* C_p = C + matrix_mma_c_row_idx * N + matrix_mma_c_col_idx;
            wmma::load_matrix_sync(c_frag, C_p, N, wmma::mem_row_major);

            // Let compiler figure out how to allocate registers for scaling
            for (uint32_t i = 0; i < c_frag.num_elements; i++)
            {
                c_frag.x[i] = alpha * c_frag.x[i] + beta * c_frag.x[i];
            }

            double* C_p2 = C + matrix_mma_c_row_idx * N + matrix_mma_c_col_idx;
            // Store the output
            wmma::store_matrix_sync(C_p2, c_frag, N, wmma::mem_row_major);

        }
    }
}