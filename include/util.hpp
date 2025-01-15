#include <iostream>
#include <vector>
#include <cmath>
#include <tuple>

#ifdef AMD
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
#endif


#ifdef NVIDIA
#define CUDA_CHECK(command)                                        \
{                                                                  \
    cudaError_t stat = (command);                                  \
    if (stat != cudaSuccess)                                       \
    {                                                              \
        std::cerr << "CUDA error: " << cudaGetErrorString(stat) << \
        " in file " << __FILE__ << ":" << __LINE__ << std::endl;   \
        exit(-1);                                                  \
    }                                                              \
}

template <typename KernelCallable>
float run_kernel_with_optional_timing(KernelCallable kernel_call, bool timed = false) {
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

        // Destroy CUDA events
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaDeviceSynchronize();

        return milliseconds;
    } else {
        // Just execute the kernel
        kernel_call();
        return 0.;
    }
}

void device_info() {
    int device_id;
    cudaGetDevice(&device_id);

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device_id);

    printf("Device ID: %d\n\
       *Number of SMs: %d\n\
       Compute Capability Major: %d\n\
       Compute Capability Minor: %d\n\
       memoryBusWidth: %d\n\
       *maxThreadsPerBlock: %d\n\
       maxThreadsPerMultiProcessor: %d\n\
       *totalGlobalMem: %zuM\n\
       sharedMemPerBlock: %zuKB\n\
       *sharedMemPerMultiprocessor: %zuKB\n\
       totalConstMem: %zuKB\n\
       *multiProcessorCount: %d\n\
       *Warp Size: %d\n",
           device_id,
           props.multiProcessorCount,
           props.major,
           props.minor,
           props.memoryBusWidth,
           props.maxThreadsPerBlock,
           props.maxThreadsPerMultiProcessor,
           props.totalGlobalMem / 1024 / 1024,
           props.sharedMemPerBlock / 1024,
           props.sharedMemPerMultiprocessor / 1024,
           props.totalConstMem / 1024,
           props.multiProcessorCount,
           props.warpSize);
}
#endif

int div_ceil(int numerator, int denominator) {
    std::div_t res = std::div(numerator, denominator);
    return res.rem ? (res.quot + 1) : res.quot;
}

size_t count_flops_gemm(int M, int N, int K) {
    return static_cast<size_t>(2) * M * N * K + (M * N) * (M * N);
}

/// @brief  Generate a Random host matrix
/// @tparam T Precision
/// @param M Number of rows
/// @param N Number of Columns
/// @return Return matrix
template <typename T>
std::vector<T> random_matrix_h(int M, int N, int seed) {

    std::mt19937 gen(seed);
    std::uniform_real_distribution<T> dist(-1, 1);
    std::vector<T> result(M * N);

    for (int i = 0; i < result.size(); ++i) {
        result[i] = dist(gen);
    }
    return result;
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

/// @brief  Perform GEMM on host device for testing
/// @tparam T
/// @tparam U
/// @param A
/// @param B
/// @param C
/// @param M
/// @param N
/// @param K
/// @param LDA
/// @param LDB
/// @param LDC
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

/// @brief Compute L2 Error between two matrices
/// @tparam T
/// @param A
/// @param B
/// @param M
/// @param N
/// @param LDA
/// @param LDB
/// @return
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
