#include "Config.h"
#ifdef NVIDIA
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_utils.hpp>
#include <gemm.cuh>
#endif

#include <gemm.hpp>
#include <utils.hpp>

int main() {

  half alpha = __float2half(1.0);
  half beta = __float2half(0.0);
  int M = 4096;
  int K = 4096;
  int N = 4096;

  half *A = new half[M * K];
  zero_init_matrix<half>(A, M * K);
  randomise_matrix<half>(A, M * K, false);

  half *B = new half[K * N];
  zero_init_matrix<half>(B, K * N);
  randomise_matrix<half>(B, K * N, false);
  auto layout = Layout::RowMajor;
  auto cache_configuration = cudaFuncCachePreferShared;

  const std::string logFile = "logFile.txt";

  std::ofstream fs;
  fs.open(logFile);

  std::ostringstream oss;
  oss << "Cache Configuration: " << cache_config_to_string(cache_configuration)
      << std::endl
      << "Data Ordering: " << ordering_to_string(layout) << std::endl;

  // Print device properties

  bool print_matrices = false;
  bool compute_error = false;

  if (print_matrices) {
    fs << "A:\n";
    print_matrix<half>(A, M, K, fs, layout);
    fs << "B:\n";
    print_matrix<half>(B, K, N, fs, layout);
  }

#ifdef NVIDIA

  device_info(fs);
  half *C = new half[M * N];
  zero_init_matrix<half>(C, M * N);

  // Copy data to device
  half *A_d, *B_d, *C_d;
  CUDA_CHECK(cudaMalloc(&A_d, (M * K) * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&B_d, (K * N) * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&C_d, (M * N) * sizeof(half)));
  CUDA_CHECK(cudaMemset(C_d, 0, M * N * sizeof(half)));
  CUDA_CHECK(
      cudaMemcpy(A_d, A, (M * K) * sizeof(half), cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(B_d, B, (K * N) * sizeof(half), cudaMemcpyHostToDevice));

  // Perform GEMM
  auto time_cublas = runKernel16(0, layout, cache_configuration, M, N, K, alpha,
                                 A_d, B_d, beta, C_d);

  auto time_kernel = runKernel16(1, layout, cache_configuration, M, N, K, alpha,
                                 A_d, B_d, beta, C_d);

    auto _gflops = performance_metrics(fs, M, N, K, time_kernel,
    time_cublas);

    // Copy back result
    CUDA_CHECK(
        cudaMemcpy(A, A_d, (M * K) * sizeof(half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(
        cudaMemcpy(B, B_d, (K * N) * sizeof(half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(
        cudaMemcpy(C, C_d, (M * N) * sizeof(half), cudaMemcpyDeviceToHost));

  //   if (compute_error) {
  //     half *C_cpu = new half[M * N];
  //     zero_init_matrix<half>(C_cpu, M * N);
  //     // runHgemmCpu(layout, M, N, K, alpha, A, B, beta, C_cpu);

  //     if (print_matrices) {
  //       fs << "C: \n";
  //       print_matrix<half>(C, M, N, fs, layout);
  //       fs << "C CPU: \n";
  //     //   print_matrix<half>(C_cpu, M, N, fs, layout);
  //     }

  //     // Test
  //     auto error = compute_relative_error_fro<half>(C, C_cpu, M, N, layout);
  //     fs << "Relative Error wrt CPU (Frobenius): " << error << std::endl;
  //     oss << "Relative Error wrt CPU (Frobeinus): " << error << std::endl;
  //   }

  // Free resources
  CUDA_CHECK(cudaFree(A_d));
  CUDA_CHECK(cudaFree(B_d));
  CUDA_CHECK(cudaFree(C_d));

#endif
}
