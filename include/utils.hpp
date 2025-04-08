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

template <typename T>
void print_matrix(const T *A, int M, int N, std::ofstream &fs) {
  int i;
  fs << std::setprecision(4)
     << std::fixed; // Set floating-point precision and fixed notation
  fs << "[";
  for (i = 0; i < M * N; i++) {
    if ((i + 1) % N == 0)
      fs << std::setw(5) << A[i]; // Set field width and write the value
    else
      fs << std::setw(5) << A[i] << ", ";
    if ((i + 1) % N == 0) {
      if (i + 1 < M * N)
        fs << ";\n";
    }
  }
  fs << "]\n";
}

template <typename T>
double compute_error_fro(T *A, T *B, int M, int N) {
    double error = 0.0;

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            double diff = A[i * N + j] - B[i * N + j];  // Correct indexing
            error += diff * diff;
        }
    }

    return std::sqrt(error);  // Take the square root at the end
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