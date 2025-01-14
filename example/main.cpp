#include <hip/hip_runtime.h>
#include <stdio.h>
#include <random>
#include <vector>
#include <iostream>

#include "helper.hpp"
#include "gemm.hpp"

int main() {

    // Allocate some host matrices

    std::mt19937 gen(0);
    std::uniform_real_distribution<double> dist(-1, 1);
    std::vector<double> A_h(A_size);

    for (int i = 0; i < A_h.size(); ++i) {
        A_h[i] = dist(gen);
    }

    std::vector<double> B_h(B_size);

    for (int i = 0; i < B_h.size(); ++i) {
        B_h[i] = dist(gen);
    }

    // Calculate reference D on host
    std::vector<double> Dref_h(D_size);

    gemm_host(A_h, B_h, Dref_h, M, N, K, LDA, LDB, LDD);

    // Make and populate device buffers
    double *A_d, *B_d, *D_d;
    HIP_CHECK(hipMalloc(&A_d, A_size * sizeof(double)));
    HIP_CHECK(hipMalloc(&B_d, B_size * sizeof(double)));
    HIP_CHECK(hipMalloc(&D_d, D_size * sizeof(double)));
    HIP_CHECK(hipMemcpy(A_d, A_h.data(), A_size * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(B_d, B_h.data(), B_size * sizeof(double), hipMemcpyHostToDevice));

    // Launch GEMM kernel
    dim3 grid(1, 1, 1);
    dim3 block(16, 4, 1);
    dgemm_16x16x16 <<<grid, block>>>(A_d, B_d, D_d);
    HIP_CHECK(hipGetLastError());

    // Copy result back to host
    std::vector<double> D_h(D_size);
    HIP_CHECK(hipMemcpy(D_h.data(), D_d, D_size * sizeof(double), hipMemcpyDeviceToHost));

    std::cout << "Sum of squared differences of host/device result matrices: "
              << compute_l2_error(Dref_h, D_h, M, N, LDD, LDD)
              << std::endl;

    HIP_CHECK(hipFree(D_d));
    HIP_CHECK(hipFree(B_d));
    HIP_CHECK(hipFree(A_d));

    return 0;
}
