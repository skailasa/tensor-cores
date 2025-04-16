#pragma once

template <const int BLOCKSIZE>
__global__ void sgemm_smem_cache_blocking_row_major(int M, int N, int K, float alpha, const float *A,
                                      const float *B, float beta, float *C);


template <const int BLOCKSIZE>
__global__ void sgemm_smem_cache_blocking_column_major(int M, int N, int K, float alpha, const float *A,
                                      const float *B, float beta, float *C);