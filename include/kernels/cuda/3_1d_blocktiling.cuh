#pragma once

template <const int BM, const int BN, const int BK, const int TM>
__global__ void sgemm_smem_1d_blocktiling_row_major(
    int M, int N, int K, float alpha, const float *A,
    const float *B, float beta, float *C);


template <const int BM, const int BN, const int BK, const int TM>
__global__ void sgemm_smem_1d_blocktiling_column_major(int M, int N, int K, float alpha, const float *A,
    const float *B, float beta, float *C);