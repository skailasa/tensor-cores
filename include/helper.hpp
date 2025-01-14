#include <iostream>
#include <vector>
#include <cmath>
#include <tuple>

// HIP error check
#define HIP_CHECK(command)                                    \
{                                                             \
  hipError_t stat = (command);                                \
  if(stat != hipSuccess)                                      \
  {                                                           \
    std::cerr << "HIP error: " << hipGetErrorString(stat) <<  \
    " in file " << __FILE__ << ":" << __LINE__ << std::endl;  \
    exit(-1);                                                 \
  }                                                           \
}

int div_ceil(int numerator, int denominator) {
    std::div_t res = std::div(numerator, denominator);
    return res.rem ? (res.quot + 1) : res.quot;
}

size_t count_flops_gemm(int M, int N, int K) {
    return static_cast<size_t>(2) * M * N * K + (M * N) * (M * N);
}


template <typename T>
std::tuple<size_t, size_t> count_memory_gemm(int M, int N, int K) {
    size_t size = sizeof(T);

    // Reads: A (M x K) and B (K x N) and C (M x N)
    size_t read = size * (M * K + K * N + M * N);

    // Write: C (M x N)
    size_t write = size * (M * N);

    return std::make_tuple(read, write);
}


template<typename T>
void print_matrix(const std::vector<T> &A,
                  const int M,
                  const int N,
                  const int LDA) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            std::cout << A[n + m * LDA] << "  ";
        }

        std::cout << std::endl;
    }
}


template<typename T>
void print_matrix_batch(const std::vector<T> &A,
                        const int M,
                        const int N,
                        const int nBatch,
                        const int LDA,
                        const int batchStride) {
    for (int b = 0; b < nBatch; ++b) {
        std::cout << "Batch " << b << ":" << std::endl;

        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                std::cout << A[n + m * LDA + b * batchStride] << "  ";
            }

            std::cout << std::endl;
        }
    }
}

template<typename T, typename U>
void gemm_host(const std::vector<U> &A,
               const std::vector<U> &B,
               std::vector<T> &C,
               const int M,
               const int N,
               const int K,
               const int LDA,
               const int LDB,
               const int LDC) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            T c = 0.0;

            for (int k = 0; k < K; ++k) {
                c += A[k + m * LDA] * B[n + k * LDB];
            }

            C[n + m * LDC] = c;
        }
    }
}

template<typename T, typename U>
void gemm_host_batch(const std::vector<U> &A,
                     const std::vector<U> &B,
                     std::vector<T> &C,
                     const int M,
                     const int N,
                     const int K,
                     const int nBatch,
                     const int LDA,
                     const int LDB,
                     const int LDC,
                     const int batchStrideA,
                     const int batchStrideB,
                     const int batchStrideC) {
    for (int b = 0; b < nBatch; ++b) {
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                T c = 0.0;

                for (int k = 0; k < K; ++k) {
                    c += A[k + m * LDA + b * batchStrideA] * B[n + k * LDB + b * batchStrideB];
                }

                C[n + m * LDC + b * batchStrideC] = c;
            }
        }
    }
}

template <typename KernelCallable>
void run_kernel_with_optional_timing_cuda(KernelCallable kernel_call, bool timed = false) {
    if (timed) {
        // Create CUDA events for timing
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Record start event
        cudaEventRecord(start, 0);

        // Execute the kernel
        kernel_call();

        // Record stop event
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        // Calculate and print elapsed time
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;

        // Destroy CUDA events
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaDeviceSynchronize();
    } else {
        // Just execute the kernel
        kernel_call();
    }
}

template<typename T>
double compute_l2_error(const std::vector<T> &A,
                        const std::vector<T> &B,
                        const int M,
                        const int N,
                        const int LDA,
                        const int LDB) {

    double err = 0.0;

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            const double x = A[n + LDA * m] - B[n + LDB * m];
            err += x * x;
        }
    }

    return err;
}

template<typename T>
double compute_l2_error_batch(const std::vector<T> &A,
                              const std::vector<T> &B,
                              const int M,
                              const int N,
                              const int nBatch,
                              const int LDA,
                              const int LDB,
                              const int batchStrideA,
                              const int batchStrideB) {

    double err = 0.0;

    for (int b = 0; b < nBatch; ++b) {
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                const double x = A[n + LDA * m + b * batchStrideA] - B[n + LDB * m + b * batchStrideB];
                err += x * x;
            }
        }
    }

    return err;
}
