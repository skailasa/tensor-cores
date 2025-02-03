#include <iostream>
#include <vector>
#include <cmath>
#include <tuple>
#include <concepts>
#include <assert.h>

#ifdef AMD

#include <hip/hip_runtime.h>


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

void device_info() {
    int device_id;
    hipGetDevice(&device_id);

    hipDeviceProp_t props;
    hipGetDeviceProperties(&props, device_id);

    printf("Device ID: %d\n\
       *Number of SMs: %d\n\
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


// A first implementation of tiled memcpy, coalesced GMEM reads, bank conflict free writes
template <std::floating_point T>
__device__ __forceinline__ void tileMemcpy(
    T* src,
    T* dst,
    const unsigned int src_stride,
    const unsigned int tile_rows,
    const unsigned int tile_cols
) {

    // Flatten out 2D grid of threads in order of increasing threadIdx.x
    const unsigned int thread_idx = threadIdx.y * blockDim.x + threadIdx.x;
    const unsigned int num_threads = blockDim.x * blockDim.y;

    // Check that number of threads is a multiple of number of columns in the tile
    assert(num_threads % tile_cols == 0);

    // assign each thread a row/column in the tile, calculate the column step
    const unsigned int row_step = num_threads / tile_cols;
    const unsigned int thread_row = thread_idx / tile_cols;
    const unsigned int thread_col = thread_idx % tile_cols;

    for (unsigned int r = thread_row; r < tile_rows; r += row_step) {
        dst[r * tile_cols + thread_col] = src[r * src_stride + thread_col];
    }

}

/// Convenience wrappers around PTX instructions
__device__ __forceinline__ uint32_t cvta_to_shared_u32(const void *pointer) {
    uint32_t address;
    asm("{\n\t"
        "  .reg .u64 u64addr;\n\t"
        "  cvta.to.shared.u64 u64addr, %1;\n\t"
        "  cvt.u32.u64 %0, u64addr;\n\t"
        "}"
        : "=r"(address)
        : "l"(pointer));
    return address;
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
