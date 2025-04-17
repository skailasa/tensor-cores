#include "Config.h"
#ifdef NVIDIA
#include <gemm.cuh>
#include <cuda_utils.hpp>
#include <cuda_runtime.h>
#endif

#include <utils.hpp>
#include <gemm.hpp>

int main() {

    float alpha = 1.0;
    float beta = 0.0;
    int M = 4096;
    int K = 4096;
    int N = 4096;

    float* A = new float[M * K];
    zero_init_matrix<float>(A, M * K);
    randomise_matrix<float>(A, M * K, false);

    float* B = new float[K * N];
    zero_init_matrix<float>(B, K * N);
    randomise_matrix<float>(B, K * N, false);
    auto layout = Layout::RowMajor;
    auto cache_configuration = cudaFuncCachePreferL1;

    const std::string logFile = "logFile.txt";

    std::ofstream fs;
    fs.open(logFile);

    std::ostringstream oss;
    oss << "Cache Configuration: " << cache_config_to_string(cache_configuration) << std::endl
        << "Data Ordering: " << ordering_to_string(layout) << std::endl;

    // Print device properties

    bool print_matrices = false;
    bool compute_error = false;

    if (print_matrices) {
        fs << "A:\n";
        print_matrix<float>(A, M, K, fs, layout);
        fs << "B:\n";
        print_matrix<float>(B, K, N, fs, layout);
    }

#ifdef NVIDIA

    device_info(fs);
    float* C = new float[M * N];
    zero_init_matrix<float>(C, M * N);

    // Copy data to device
    float *A_d, *B_d, *C_d;
    CUDA_CHECK(cudaMalloc(&A_d, (M * K) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&B_d, (K * N) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&C_d, (M * N) * sizeof(float)));
    CUDA_CHECK(cudaMemset(C_d, 0, M * N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(A_d, A, (M * K) * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B, (K * N) * sizeof(float), cudaMemcpyHostToDevice));

    // Perform GEMM
    auto time_cublas = runKernel32(0, layout, cache_configuration, M, N, K, alpha, A_d, B_d, beta, C_d);
    auto time_kernel = runKernel32(10, layout, cache_configuration, M, N, K, alpha, A_d, B_d, beta, C_d);

    auto _gflops = performance_metrics(fs, M, N, K, time_kernel, time_cublas);

    // Copy back result
    CUDA_CHECK(cudaMemcpy(A, A_d, (M * K) * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(B, B_d, (K * N) * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(C, C_d, (M * N) * sizeof(float), cudaMemcpyDeviceToHost));

    if (compute_error) {
        float* C_cpu = new float[M * N];
        zero_init_matrix<float>(C_cpu, M * N);
        runSgemmCpu(layout, M, N, K, alpha, A, B, beta, C_cpu);

        if (print_matrices) {
            fs << "C: \n";
            print_matrix<float>(C, M, N, fs, layout);
            fs << "C CPU: \n";
            print_matrix<float>(C_cpu, M, N, fs, layout);
        }

        // Test
        auto error = compute_relative_error_fro<float>(C, C_cpu, M, N, layout);
        fs << "Relative Error wrt CPU (Frobenius): " << error << std::endl;
        oss << "Relative Error wrt CPU (Frobeinus): " << error << std::endl;
    }

    // Free resources
    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFree(C_d));


#endif

}
