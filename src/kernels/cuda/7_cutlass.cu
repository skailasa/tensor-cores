#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>

__global__ void sgemm_cutlass(int M, int N, int K, float alpha, const float *A,
                              const float *B, float beta, float *C) {
}