constexpr int M = 16;
constexpr int N = 16;
constexpr int K = 4;

// Kernel that performs a simple computation
__global__ void compute_kernel(int *result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx == 0) {
        *result = 42; // Arbitrary value to indicate successful execution
    }
}

__global__ void sgemm_16x16x4(const float *A, const float *B, float *D) {
    using float4 = __attribute__( (__vector_size__(K * sizeof(float)) )) float;
    float4 dmn = {0};

    int mk = threadIdx.y + K * threadIdx.x;
    int kn = threadIdx.x + N * threadIdx.y;

    float amk = A[mk];
    float bkn = B[kn];
    dmn = __builtin_amdgcn_mfma_f32_16x16x4f32(amk, bkn, dmn, 0, 0, 0);

    for (int i = 0; i < 4; ++i) {
        const int idx = threadIdx.x + i * N + threadIdx.y * 4 * N;
        D[idx] = dmn[i];
    }
}
