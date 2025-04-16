#pragma once

#include "types.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <cmath>

#define PRINT_FUNC_NAME(func) std::cout << #func << std::endl;

// Declarations of non-template functions
int ceil_div(int numerator, int denominator);
double count_flops(int M, int N, int K);

std::string ordering_to_string(Layout layout);


double performance_metrics(std::ofstream& fs, int M, int N, int K, float time_kernel, float time_cublas);

// Template function definitions (define here, since templates must be visible to all TUs)
template <typename T>
constexpr int count_memory(int M, int N, int K) {
    size_t size = sizeof(T);
    return size * (M * K + K * N + 2 * M * N);
}

template <typename T>
void randomise_matrix(T *mat, int size, bool seeded) {
    if (!seeded) {
        srand(time(nullptr));
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
void print_matrix(const T* A, int M, int N, std::ofstream& fs, Layout layout) {
    fs << std::setprecision(4) << std::fixed;
    fs << "[";
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            int idx = (layout == Layout::RowMajor) ? i * N + j : j * M + i;
            fs << std::setw(8) << A[idx];
            if (j < N - 1) fs << ", ";
        }
        if (i < M - 1) fs << ";\n";
    }
    fs << "]\n";
}

template <typename T>
double compute_frobenius_norm(const T* A, int M, int N, Layout layout) {
    double norm = 0.0;
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            int idx = (layout == Layout::RowMajor) ? (i * N + j) : (j * M + i);
            norm += static_cast<double>(A[idx]) * static_cast<double>(A[idx]);
        }
    return std::sqrt(norm);
}

template <typename T>
double compute_relative_error_fro(const T* A, const T* B, int M, int N, Layout layout) {
    double diff_norm = 0.0, ref_norm = 0.0;
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            int idx = (layout == Layout::RowMajor) ? (i * N + j) : (j * M + i);
            double a = static_cast<double>(A[idx]);
            double b = static_cast<double>(B[idx]);
            diff_norm += (a - b) * (a - b);
            ref_norm += a * a;
        }
    return std::sqrt(diff_norm) / std::sqrt(ref_norm + 1e-20);
}
