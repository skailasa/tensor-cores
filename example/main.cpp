#include <hip/hip_runtime.h>
#include <stdio.h>

// Kernel that performs a simple computation
__global__ void compute_kernel(int* result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        *result = 42; // Arbitrary value to indicate successful execution
    }
}

int main() {
    int h_result = 0;   // Host result
    int* d_result;      // Device result

    // Allocate memory on the device
    hipMalloc((void**)&d_result, sizeof(int));

    // Initialize device memory
    hipMemset(d_result, 0, sizeof(int));

    // Launch the kernel with 1 block and 1 thread
    hipLaunchKernelGGL(compute_kernel,
                       dim3(1),     // 1 block
                       dim3(1),     // 1 thread
                       0,           // Shared memory
                       0,           // Stream
                       d_result);   // Pass device pointer

    // Copy the result back to the host
    hipMemcpy(&h_result, d_result, sizeof(int), hipMemcpyDeviceToHost);

    // Free device memory
    hipFree(d_result);

    // Print the result
    printf("Result from GPU: %d\n", h_result);

    return 0;
}
