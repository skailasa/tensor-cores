#pragma once
#include <cuda_fp16.h>

template<const int BM, const int BN, const int BK, const int WM, const int WN, const int WK, const int NUM_THREADS> __global__ void __launch_bounds__(NUM_THREADS) hgemm_warptiling_tensor_cores(int M, int N, int K, half alpha, half *A, half *B,
    half beta, half *C);