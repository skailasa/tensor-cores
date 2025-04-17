#include "gemm.hpp"
#include "omp.h"
#include <types.hpp>

void runSgemmCpu(Layout layout, int M, int N, int K, float alpha, float *A,
                 float *B, float beta, float *C) {

  // Perform matrix multiplication
  if (layout == Layout::RowMajor) {
// C[i * N + j] = alpha * A[i * K + k] * B[k * N + j] + beta * C[i * N + j]
#pragma omp parallel for collapse(1)
    for (int row = 0; row < M; row++) {
      for (int col = 0; col < N; col++) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
          sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
      }
    }
  } else if (layout == Layout::ColumnMajor) {
// C[i + j * M] = alpha * sum(A[i + k * M] * B[k + j * K]) + beta * C[i + j * M]
#pragma omp parallel for collapse(1)
    for (int col = 0; col < N; col++) {   // column of C and B
      for (int row = 0; row < M; row++) { // row of C and A
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
          sum += A[k * M + row] * B[col * K + k];
        }
        C[col * M + row] = alpha * sum + beta * C[col * M + row];
      }
    }
  }
}