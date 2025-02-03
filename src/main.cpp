#include <cstdlib>
#include <stdio.h>
#include <random>
#include <vector>
#include <iostream>
#include "util.hpp"
#include "gemm.hpp"

int main() {

    bool timed = true;

    // Allocate some host matrices
    int M = 4096;
    int N = 4096;
    int K = 4096;

    float alpha = 1.0;
    float beta = 0.0;

    auto A_h = random_matrix_h<float>(M, K, 0);
    auto B_h = random_matrix_h<float>(K, N, 0);

    // Calculate reference C on host
    std::vector<float> C_ref_h(M * N);

    device_info();

#ifdef AMD

    // // Make and populate device buffers
    // float *A_d, *B_d, *D_d;
    // HIP_CHECK(hipMalloc(&A_d, A_size * sizeof(float)));
    // HIP_CHECK(hipMalloc(&B_d, B_size * sizeof(float)));
    // HIP_CHECK(hipMalloc(&D_d, D_size * sizeof(float)));
    // HIP_CHECK(hipMemcpy(A_d, A_h.data(), A_size * sizeof(float), hipMemcpyHostToDevice));
    // HIP_CHECK(hipMemcpy(B_d, B_h.data(), B_size * sizeof(float), hipMemcpyHostToDevice));

    // // Launch GEMM kernel
    // dim3 grid(1, 1, 1);
    // dim3 block(16, 4, 1);
    // dgemm_16x16x16 <<<grid, block>>>(A_d, B_d, D_d);
    // HIP_CHECK(hipGetLastError());

    // // Copy result back to host
    // std::vector<float> D_h(D_size);
    // HIP_CHECK(hipMemcpy(D_h.data(), D_d, D_size * sizeof(float), hipMemcpyDeviceToHost));

    // std::cout << "Sum of squared differences of host/device result matrices: "
    //         << compute_l2_error(Dref_h, D_h, M, N, LDD, LDD)
    //         << std::endl;

    // HIP_CHECK(hipFree(D_d));
    // HIP_CHECK(hipFree(B_d));
    // HIP_CHECK(hipFree(A_d));

#endif

#ifdef NVIDIA

    // // Allocate space for result
    // std::vector<float> C_h(M * N);

    // // Copy data to device
    // float *A_d, *B_d, *C_d;
    // CUDA_CHECK(cudaMalloc(&A_d, (M * K) * sizeof(float)));
    // CUDA_CHECK(cudaMalloc(&B_d, (K * N) * sizeof(float)));
    // CUDA_CHECK(cudaMalloc(&C_d, (M * N) * sizeof(float)));
    // CUDA_CHECK(cudaMemcpy(A_d, A_h.data(), (M * K) * sizeof(float), cudaMemcpyHostToDevice));
    // CUDA_CHECK(cudaMemcpy(B_d, B_h.data(), (K * N) * sizeof(float), cudaMemcpyHostToDevice));

    // // Launch kernel
    // dim3 grid(div_ceil(M, 32), div_ceil(N, 32), 1);
    // dim3 block(32, 32, 1);

    // sgemm(1, M, N, K, alpha, A_d, B_d, beta, C_d, grid, block, timed);

    // // Copy back result
    // CUDA_CHECK(cudaMemcpy(C_h.data(), C_d, (M * N) * sizeof(float), cudaMemcpyDeviceToHost));

    // // Free resources
    // CUDA_CHECK(cudaFree(A_d));
    // CUDA_CHECK(cudaFree(B_d));
    // CUDA_CHECK(cudaFree(C_d));

    // Allocate space for result
    std::vector<double> C_h(M * N);

    // Copy data to device
    double *A_d, *B_d, *C_d;
    CUDA_CHECK(cudaMalloc(&A_d, (M * K) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&B_d, (K * N) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&C_d, (M * N) * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(A_d, A_h.data(), (M * K) * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B_h.data(), (K * N) * sizeof(double), cudaMemcpyHostToDevice));

    // Launch kernel
    dim3 grid(div_ceil(M, 32), div_ceil(N, 32), 1);
    dim3 block(32, 32, 1);

    dgemm(1, M, N, K, alpha, A_d, B_d, beta, C_d, grid, block, timed);

    // Copy back result
    CUDA_CHECK(cudaMemcpy(C_h.data(), C_d, (M * N) * sizeof(double), cudaMemcpyDeviceToHost));

    // Free resources
    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFree(C_d));
#endif

    return 0;
}
