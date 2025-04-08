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

void device_info(std::ofstream& fs) {
    int device_id;
    cudaGetDevice(&device_id);

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device_id);

    std::ostringstream oss;

    oss << "Device Info \n"
        << "-------------- \n"
        << "Device ID: " << device_id << "\n"
        << "*Number of SMs: " << props.multiProcessorCount << "\n"
        << "Compute Capability: " << props.major << "." << props.minor << "\n"
        << "memoryBusWidth: " << props.memoryBusWidth << " bits\n"
        << "*maxThreadsPerBlock: " << props.maxThreadsPerBlock << "\n"
        << "maxThreadsPerMultiProcessor: " << props.maxThreadsPerMultiProcessor << "\n"
        << "*totalGlobalMem: " << props.totalGlobalMem / (1024 * 1024) << " MB\n"
        << "sharedMemPerBlock: " << props.sharedMemPerBlock / 1024 << " KB\n"
        << "*sharedMemPerMultiprocessor: " << props.sharedMemPerMultiprocessor / 1024 << " KB\n"
        << "totalConstMem: " << props.totalConstMem / 1024 << " KB\n"
        << "*multiProcessorCount: " << props.multiProcessorCount << "\n"
        << "*Warp Size: " << props.warpSize << "\n\n";

    // Print to stdout
    std::cout << oss.str();

    // Write to file
    fs << oss.str();
}

std::string cache_config_to_string(cudaFuncCache config) {
    switch (config) {
    case cudaFuncCachePreferNone:
        return "cudaFuncCachePreferNone";
    case cudaFuncCachePreferShared:
        return "cudaFuncCachePreferShared";
    case cudaFuncCachePreferL1:
        return "cudaFuncCachePreferL1";
    case cudaFuncCachePreferEqual:
        return "cudaFuncCachePreferEqual";
    default:
        return "Unknown cudaFuncCache value";
    }
}

std::string ordering_to_string(Layout layout) {
    switch (layout) {
    case Layout::RowMajor:
        return "Row Major";
    case Layout::ColumnMajor:
        return "Column Major";
    default:
        return "Unknown Ordering";
    }
}


double performance_metrics(std::ofstream& fs, int M, int N, int K, float time_kernel, float time_cublas) {
    std::ostringstream oss;

    auto gflops =  count_flops(M, N, K) / 1000000000.0;
    oss << "Performance Metrics" << std::endl
        << "-------------------" << std::endl
        << "FLOP: " << gflops << " GFLOP" << std::endl
        << "Throughput: " << gflops / (time_kernel / 1e3) << " GFLOP/s" << std::endl
        << "Throughput cuBLAS: " << gflops / (time_cublas / 1e3) << " GFLOP/s" << std::endl
        << "Throughput (% of cuBLAS): " << time_cublas/time_kernel * 100.0 << "  %" << std::endl
        << "Time (kernel): " <<  time_kernel << " ms" << std::endl
        << "Time (cuBLAS): " <<  time_cublas << " ms" << std::endl;

    // Print to stdout
    std::cout << oss.str();

    // Write to file
    fs << oss.str();

    return gflops;
}



#endif