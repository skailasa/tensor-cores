#pragma once

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemm_vectorise_smem_and_gmem_accesses_row_major(
    int M, int N, int K, float alpha, const float *A,
    const float *B, float beta, float *C);