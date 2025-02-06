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
            tmp += A[x * K + i] * B[i * N + y];
        }

        // C = alpha * (A@B) + beta * C
        C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    }
}

__global__ void dgemm_simple(int M, int N, int K, double alpha, const double *A, const double *B,
                             double beta, double *C) {

    // Position in C that this thread is responsible for
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    // Ensure thread is within bounds
    if (x < M && y < N) {
        double tmp = 0.0;

        for (int i = 0; i < K; ++i) {
            tmp += A[x * K + i] * B[i * N + y];
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
__global__ void dgemm_wmma(const int M, int N, int K,
                           double alpha, const double *A,
                           const double *B, double beta, double *C) {

    // Leading dimension, row major
    int lda = K;
    int ldb = N;
    int ldc = N;

    // Tile using a 2D grid
    int warp_m = (blockIdx.y * blockDim.y + threadIdx.y); // warp row
    int warp_n = (blockIdx.x * blockDim.x + threadIdx.x) / 32; // warp column

    // Compute starting row and column for the tile of C that this thread is contributing to
    int c_row = warp_m * 8;
    int c_col = warp_n * 8;

    // Declare the accumulator fragment and initialize it to 0.
    wmma::fragment<wmma::accumulator, 8, 8, 4, double> acc_frag;
    wmma::fragment<wmma::accumulator, 8, 8, 4, double> c_frag;
    wmma::fill_fragment(acc_frag, 0.0);
    wmma::fill_fragment(c_frag, 0.0);

    // initialise other fragments
    wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::row_major> b_frag;

    // loop over shared dimension
    for (int k = 0; k < K ; k += 4) {

        int a_row = warp_m * 8;
        int a_col = k;
        int b_row = k;
        int b_col = warp_n * 8;

        // bounds checking
        if (a_row < M && a_col < K && b_row < K && b_col < N) {
            // load 8x4 tiles of A (row major)
            const double *A_tile = A + a_row * lda + a_col;
            const double *B_tile = B + b_row * ldb + b_col;

            wmma::load_matrix_sync(a_frag, A_tile, lda);
            wmma::load_matrix_sync(b_frag, B_tile, ldb);

            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    if (c_row < M && c_col < N) {
        double *C_tile = C + c_row * ldc + c_col;
        wmma::load_matrix_sync(c_frag, C_tile, ldc, wmma::mem_row_major);

        for (int i=0; i < c_frag.num_elements; i++) {
            c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
        }

        wmma::store_matrix_sync(C_tile, c_frag, ldc, wmma::mem_row_major);
    }

}

