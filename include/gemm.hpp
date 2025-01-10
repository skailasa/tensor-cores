#include <hip/hip_runtime.h>

constexpr int M = 16;
constexpr int N = 16;
constexpr int K = 16;

constexpr int LDA = K;
constexpr int LDB = N;
constexpr int LDD = N;

constexpr int A_size = M * LDA;
constexpr int B_size = K * LDB;
constexpr int D_size = M * LDD;

__global__ void dgemm_16x16x16(const double *A, const double *B, double *D) {

#if __gfx90a__
    // This kernel computes a 16x16x16 matrix multiplication using a single wavefront.
    using double4 = __attribute__((__vector_size__(4 * sizeof(double)))) double;
    double4 d = {0}; // zero out 4 * 2 vanilla VGPRs

    /*
    One invocation of v_mfma_f64_16x16x4f64 accumulates the sum of four outer products,
    four columns of A with four rows of B, into result matrix D (which is in AccVGPRs).
    So we need 4 iterations to compute the full matrix D, starting with the leftmost four
    columns of A and the topmost four colums of B, and then moving four columns to the right
    for A, and down for B, for every iteration.

    For both the four columns of A, and the four rows of B, we use a single regular VGPR pair.
    With 64 lanes, that covers the 64 values for the four rows/columns of 16 items each.
    For the four A columns: lanes 0-15 cover the 1st column, ..., lanes 48-63 cover the 4th column.
    For the four B rows: lanes 0-15 cover the 1st row, ..., lanes 48-63 cover the 4th row.
    Note that A and B are in row-major order.

    This kernel is called with a single wavefront in dim3(16, 4) layout
    */

    int a_idx = LDA * threadIdx.x + threadIdx.y;
    int b_idx = threadIdx.x + LDB * threadIdx.y;

    for (int i = 0; i < 4; ++i) {
        const double a = A[a_idx];
        const double b = B[b_idx];

        d = __builtin_amdgcn_mfma_f64_16x16x4f64(a, b, d, 0, 0, 0);
        //                                       ^  ^  ^
        //D(=C)                                  |  |  C(=D)
        //                    two columns of A---|  |--- two rows of B
        a_idx += 4;     // move two columns to the right
        b_idx += 4 * LDB; // move two rows down
    }

    /*
    For v_mfma_f64_16x16x4f64, the layout of rows 0-3, 4-7, 8-11, and 12-15 of the
    matrices D (and C) is the same as the layout for B; see above
    */
    for (int i = 0; i < 4; ++i) {
        const int d_idx =  threadIdx.x           // consecutive threads cover 16 consecutive columns
                           + 4 * LDD * i          // consecutive registers skip 4 rows
                           + LDD * threadIdx.y;   // groups of 16 lanes cover consecutive rows
        D[d_idx] = d[i];
    }

#endif
}