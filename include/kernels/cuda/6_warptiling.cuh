#pragma once

// template <const int BM, const int BN, const int BK, const int WM, const int WN,
//           const int WNITER, const int TM, const int TN, const int NUM_THREADS>
// __global__ void __launch_bounds__(NUM_THREADS)
//     sgemm_warptiling(int M, int N, int K, float alpha, float *A, float *B,
//                     float beta, float *C);



template<const int BM, const int BN, const int BK, const int WM, const int WN> __global__ void sgemm_warptiling(int M, int N, int K, float alpha, float *A, float *B,
    float beta, float *C);