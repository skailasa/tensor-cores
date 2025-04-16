#pragma once

#include <cublas.h>
#include <types.hpp>

// Type alias for GEMM kernels i.e. pointers to kernel functions
// equivalent to typedef void (*my_kernel)(arg1, arg2,...) - a raw pointer to a function
// can point to any function with this/that signature
using KernelPtr = void(*)(int, int, int, float, const float*, const float*, float, float*);

/// @brief  Call cubLAS with single precision inputs
void runCublasF32(cublasHandle_t handle, Layout layout, int M, int N, int K, float alpha, float *A, float *B, float beta, float*C);

/// @brief  Call cubLAS with single precision inputs casted down to BF16 for the actual mul
void runCublasB16(cublasHandle_t handle, Layout layout, int M, int N, int K, float alpha, float *A, float *B, float beta, float*C);

/// @brief  Call cubLAS with single precision inputs casted to TF32 for the actual mul
void runCublasT32(cublasHandle_t handle, Layout layout, int M, int N, int K, float alpha, float *A, float *B, float beta, float*C);

/// @brief  Call cubLAS with single precision inputs casted down to F16 for the actual mul
void runCublasF16(cublasHandle_t handle, Layout layout, int M, int N, int K, float alpha, float *A, float *B, float beta, float*C);

void runSGemmNaive(Layout layout, cudaFuncCache cache_configuration,  int M, int N, int K, float alpha, float *A, float *B, float beta, float *C);

void runSgemmSharedMemCacheBlocking(Layout layout, cudaFuncCache cache_configuration, int M, int N, int K, float alpha, float *A, float *B, float beta, float *C);

void runSgemm1dBlockTiling(Layout layout, cudaFuncCache cache_configuration, int M, int N, int K, float alpha, float *A, float *B, float beta, float *C);

void runSgemm2dBlockTiling(Layout layout, cudaFuncCache cache_configuration, int M, int N, int K, float alpha, float *A, float *B, float beta, float *C);

float runKernel32(int kernel_number, Layout layout, cudaFuncCache cache_configuration, int M, int N, int K, float alpha, float *A, float *B, float beta, float *C);