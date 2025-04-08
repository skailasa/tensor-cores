#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>

constexpr int ceil_div(int numerator, int denominator) {
    return (numerator + denominator - 1) / denominator;
}

constexpr double count_flops(int M, int N, int K) {
    return static_cast<double>(M) * N * K * 2.0;
}

template <typename T>
constexpr int count_memory(int M, int N, int K) {
    size_t size = sizeof(T);
    return size * (M * K + K * N + 2 * M * N);
}


template <typename T>
void randomise_matrix(T *mat, int size, bool seeded) {

    if (!seeded) {
        srand(time(nullptr));  // Seed only once
        seeded = true;
    }

    for (int i = 0; i < size; i++) {
        T tmp = (T)(5.0 * ((T)rand() / RAND_MAX) + 0.01 * (rand() % 100));
        tmp = (rand() % 2 == 0) ? tmp : -tmp;
        mat[i] = tmp;
    }
}

template <typename T>
void zero_init_matrix(T *mat, int size) {
  for (int i = 0; i < size; i++) {
    mat[i] = 0.0;
  }
}

enum class Layout { RowMajor, ColumnMajor };


template <typename T>
void print_matrix(const T* A, int M, int N, std::ofstream& fs, Layout layout) {
  fs << std::setprecision(4) << std::fixed;  // Set floating-point precision
  fs << "[";

  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      int idx;
      if (layout == Layout::RowMajor)
        idx = i * N + j;        // A[i][j]
      else
        idx = j * M + i;        // A(i,j) in column-major layout

      fs << std::setw(8) << A[idx];
      if (j < N - 1)
        fs << ", ";
    }

    if (i < M - 1)
      fs << ";\n";
  }

  fs << "]\n";
}

template <typename T>
double compute_frobenius_norm(const T* A, int M, int N, Layout layout) {
    double norm = 0.0;

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int idx = (layout == Layout::RowMajor) ? (i * N + j) : (j * M + i);
            norm += static_cast<double>(A[idx]) * static_cast<double>(A[idx]);
        }
    }

    return std::sqrt(norm);
}

template <typename T>
double compute_relative_error_fro(const T* A, const T* B, int M, int N, Layout layout) {
    double diff_norm = 0.0;
    double ref_norm  = 0.0;

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int idx = (layout == Layout::RowMajor) ? (i * N + j) : (j * M + i);

            double a = static_cast<double>(A[idx]);
            double b = static_cast<double>(B[idx]);

            double diff = a - b;
            diff_norm += diff * diff;
            ref_norm += a * a;
        }
    }

    return std::sqrt(diff_norm) / std::sqrt(ref_norm + 1e-20); // avoid div-by-zero
}

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
        cudaDeviceSynchronize();
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