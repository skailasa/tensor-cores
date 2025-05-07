#include "kernels.hpp"
#include "types.hpp"
#include <stdio.h>

#define  assert( X ) if ( !(X) ) \
    printf("tid %d: %s, %d\n", threadIdx.x, __FILE__, __LINE__);\
    return ;


extern __shared__ float shmem [];


/// Test that two matrices held in shared memory are the same
template <const int M, const int N> __device__ void test_matrix_equality_smem(float* A, float* B, Layout layout) {

  int threadsPerBlock = blockDim.x;

  if (layout == Layout::ColumnMajor) {
    for (int i = threadIdx.x; i < M * N; i += threadsPerBlock) {
      int row = i / N;
      int col = i % N;
      float a = A[col * M + row];
      float b = B[col * M + row];
      assert(fabsf(a - b) < 1e-4f);
    }
  } else if (layout == Layout::RowMajor) {
    for (int i = threadIdx.x; i < M * N; i += threadsPerBlock) {
      int row = i / N;
      int col = i % N;
      float a = A[row * N + col];
      float b = B[row * N + col];
      assert(fabsf(a - b) < 1e-4f);
      // assert(0);
    }
  }
}

template <const int BM, const int BN, const int BK> __device__ __forceinline__ void load_from_gmem_old(int M, int N, int K, float* A, float* As, float* As_T, float* B, float* Bs, int bk) {

  int threadId = threadIdx.x;
  int threadsPerBlock = blockDim.x;

  for (int i = threadId; i < BM * BK; i += threadsPerBlock) {
    int row = i / BK;
    int col = i % BK;
    int globalRow = blockIdx.y * BM + row;
    int globalCol = bk + col;

    if (globalRow < M && globalCol < K)
      As[row * BK + col] = A[globalRow * K + globalCol];
    else
      As[row * BK + col] = 0.0f;
  }

  // Load tile of B into shared memory
  for (int i = threadId; i < BK * BN; i += threadsPerBlock) {
    int row = i / BN;
    int col = i % BN;
    int globalRow = bk + row;
    int globalCol = blockIdx.x * BN + col;

    if (globalRow < K && globalCol < N)
      Bs[row * BN + col] = B[globalRow * N + globalCol];
    else
      Bs[row * BN + col] = 0.0f;
  }

  // Now need to transpose As
  for (int i = threadId; i < BM * BK; i += threadsPerBlock) {

    // Assign threads to physical rows and columns again assuming row major order of data
    // i.e. adjacent data correspond to adjacent threads
    int row = i / BK;
    int col = i % BK;
    int globalRow = blockIdx.y * BM + row; // convert block index into units of threads, and add local row
    int globalCol = bk + col;

    if (globalRow < M && globalCol < K)
      As_T[col * BM + row] = As[row * BK + col];
  }


  __syncthreads();
}


template <const int BM, const int BN, const int BK> __device__ __forceinline__ void load_from_gmem(int M, int N, int K, float* A, float* As, float* B, float* Bs, int bk) {

    const int threadId = threadIdx.x;
    const int threadsPerBlock = blockDim.x;

      // Load A into As, assume row major order
      for (int i = threadId; i < ((BM * BK) / 4); i += threadsPerBlock) {
        // Basically indexing by saying that number of cols 4x smaller as we read in 4 floats at a time
        // of row major data. The number of rows is still the same, and therefore so is the code to
        // calculate the physical row index
        // For future
        int row = i / BK;
        int col = i % (BK / 4);
        int globalRow = blockIdx.y * BM + row; // convert block index into units of threads, and add local row
        int globalCol = bk + col * 4;

        float4 tmp = {0.0f, 0.0f, 0.0f, 0.0f};

        if (globalRow < M && globalCol + 3 < K) {
          tmp = reinterpret_cast<float4 *>(&A[globalRow * K + globalCol])[0];
        }

        // N.B For future me, this commented out code basically reads into rows of As
        // Reading in in row major order
        // As[row * BK + col * 4]     = tmp.x;
        // As[row * BK + col * 4 + 1] = tmp.y;
        // As[row * BK + col * 4 + 2] = tmp.z;
        // As[row * BK + col * 4 + 3] = tmp.w;

        // However, to read in and transpose at the same time can switch into column major order
        // Read in and transpose at the same time
        As[(col * 4 + 0) * BM + row] = tmp.x;
        As[(col * 4 + 1) * BM + row] = tmp.y;
        As[(col * 4 + 2) * BM + row] = tmp.z;
        As[(col * 4 + 3) * BM + row] = tmp.w;
      }

      // // Load B into Bs
      for (int i = threadId; i < ((BK * BN) / 4); i += threadsPerBlock) {
        int row = i / BN;
        int col = i % (BN / 4);
        int globalRow = bk + row;
        int globalCol = blockIdx.x * BN + col * 4;

        float4 tmp = {0.0f, 0.0f, 0.0f, 0.0f};

        if (globalRow < K && globalCol + 3 < N) {
          tmp = reinterpret_cast<float4 *>(&B[globalRow * N + globalCol])[0];
        }

        Bs[row * BN + col * 4]     = tmp.x;
        Bs[row * BN + col * 4 + 1] = tmp.y;
        Bs[row * BN + col * 4 + 2] = tmp.z;
        Bs[row * BN + col * 4 + 3] = tmp.w;
      }

      __syncthreads();
}


// Compute a warp matrix multiply in shared memory
template <const int BM, const int BN, const int BK, const int WM, const int WN> __device__ __forceinline__ void process_in_smem(int M, int N, int K) {



}

template <const int BM, const int BN, const int BK> __device__ __forceinline__ void load_from_gmem_test(int M, int N, int K, float* A, float* As, float* B, float* Bs, bool test) {

    const int threadId = threadIdx.x;
    const int threadsPerBlock = blockDim.x;

    // Used for testing purposes only, if small shouldn't impact kernel performance
    // and can be commented out
    __shared__ float As_test_1[BM*BK];
    __shared__ float As_test_2[BM*BK];
    __shared__ float As_T_test[BK*BM];

    for (int bk = 0; bk < K; bk += BK) {

      if (test) {
        // A straightforward way of loading A into As, for testing.
        for (int i = threadId; i < BM * BK; i += threadsPerBlock) {
          // Assign threads to physical rows and columns again assuming row major order of data
          // i.e. adjacent data correspond to adjacent threads
          int row = i / BK;
          int col = i % BK;
          int globalRow = blockIdx.y * BM + row; // convert block index into units of threads, and add local row
          int globalCol = bk + col;

          if (globalRow < M && globalCol < K)
          As_test_1[row * BK + col] = A[globalRow*K + globalCol];
          else
          As_test_1[row * BK + col] = 0.0f;
        }
      }

      // Load A into As, assume row major order
      for (int i = threadId; i < (BM * BK / 4); i += threadsPerBlock) {

        // Basically indexing by saying that number of cols 4x smaller as we read in 4 floats at a time
        // of row major data. The number of rows is still the same, and therefore so is the code to
        // calculate the physical row index
        // For future
        int row = i / BK;
        int col = i % (BK / 4);
        int globalRow = blockIdx.y * BM + row; // convert block index into units of threads, and add local row
        int globalCol = bk + col * 4;

        float4 tmp = {0.0f, 0.0f, 0.0f, 0.0f};

        if (globalRow < M && globalCol + 3 < K) {
          tmp = reinterpret_cast<float4 *>(&A[globalRow * K + globalCol])[0];
        }

        // N.B For future me, this commented out code basically reads into rows of As
        // Reading in in row major order
        // As[row * BK + col * 4]     = tmp.x;
        // As[row * BK + col * 4 + 1] = tmp.y;
        // As[row * BK + col * 4 + 2] = tmp.z;
        // As[row * BK + col * 4 + 3] = tmp.w;
        if (test) {
          As_test_2[row * BK + col * 4]     = tmp.x;
          As_test_2[row * BK + col * 4 + 1] = tmp.y;
          As_test_2[row * BK + col * 4 + 2] = tmp.z;
          As_test_2[row * BK + col * 4 + 3] = tmp.w;
        }

        // However, to read in and transpose at the same time can switch into column major order
        // Read in and transpose at the same time
        As[(col * 4 + 0) * BM + row] = tmp.x;
        As[(col * 4 + 1) * BM + row] = tmp.y;
        As[(col * 4 + 2) * BM + row] = tmp.z;
        As[(col * 4 + 3) * BM + row] = tmp.w;
      }

      // Load B into Bs
      for (int i = threadId; i < ((BK * BN) / 4); i += threadsPerBlock) {
        int row = i / BN;
        int col = i % (BN / 4);
        int globalRow = bk + row;
        int globalCol = blockIdx.x * BN + col * 4;

        float4 tmp = {0.0f, 0.0f, 0.0f, 0.0f};

        if (globalRow < K && globalCol + 3 < N) {
          tmp = reinterpret_cast<float4 *>(&B[globalRow * N + globalCol])[0];
        }

        Bs[row * BN + col * 4]     = tmp.x;
        Bs[row * BN + col * 4 + 1] = tmp.y;
        Bs[row * BN + col * 4 + 2] = tmp.z;
        Bs[row * BN + col * 4 + 3] = tmp.w;
      }


      if (test) {
        // Then need to test how long it takes to transpose As, already in shared memory
        // This is a lot slower than reading in and transposing at the same time.
        for (int i = threadId; i < BM * BK; i += threadsPerBlock) {

          // Assign threads to physical rows and columns again assuming row major order of data
          // i.e. adjacent data correspond to adjacent threads
          int row = i / BK;
          int col = i % BK;
          int globalRow = blockIdx.y * BM + row; // convert block index into units of threads, and add local row
          int globalCol = bk + col;

          if (globalRow < M && globalCol < K)
            As_T_test[col * BM + row] = As_test_1[row * BK + col];
        }
      }

      __syncthreads();

      if (test) {
        // Test transposition
        test_matrix_equality_smem<BM, BK>(As, As_T_test, Layout::ColumnMajor);

        // Test loading
        test_matrix_equality_smem<BM, BK>(As_test_2, As_test_1, Layout::RowMajor);
      }
    }
}


template <const int BM, const int BN, const int BK, const int WM, const int WN> __global__ void sgemm_warptiling(int M, int N, int K, float alpha, float *A, float *B,
  float beta, float *C) {

    __shared__ float As[BM*BK];
    __shared__ float As_T[BM*BK];
    __shared__ float Bs[BK*BN];

    // Inner loop is unrolled, as usual.
    // The two outer loops are over thread blocks
    const int threadId = threadIdx.x;
    const int threadsPerBlock = blockDim.x;


    // Uncomment to test GMEM -> SMEM loading strategy
    // load_from_gmem_test<BM, BN, BK>(M, N, K, A, As, B, Bs, true);

    for (int bk = 0; bk < K; bk += BK) {
      load_from_gmem<BM, BN, BK>(M, N, K, A, As, B, Bs, bk);

      // Uncomment to benchmark old 2D blocktiling approach + manual transpose
      // load_from_gmem_old<BM, BN, BK>(M, N, K, A, As, As_T, B, Bs, bk);

      process_in_smem<BM, BN, BK, WM, WN>(M, N, K);

      // Force usage
      if (threadIdx.x == 0 && blockIdx.x == 0) {
        volatile float sink = 0.0f;
        sink += As[0];     // forces compiler to preserve As[]
        sink += Bs[0];     // same for Bs[]
      }

    }
}

template __global__ void sgemm_warptiling<64, 64, 64, 16, 16>(
    int, int, int, float, float*, float*, float, float*);

  template __global__ void sgemm_warptiling<64, 64, 32, 16, 16>(
    int, int, int, float, float*, float*, float, float*);

template __global__ void sgemm_warptiling<64, 64, 32, 32, 32>(
    int, int, int, float, float*, float*, float, float*);

template __global__ void sgemm_warptiling<64, 64, 64, 32, 32>(
    int, int, int, float, float*, float*, float, float*);
