#include "utils.hpp"
#include <cstdio>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <types.hpp>

int ceil_div(int numerator, int denominator) {
    return (numerator + denominator - 1) / denominator;
}

double count_flops(int M, int N, int K) {
    return static_cast<double>(M) * N * K * 2.0;
}

std::string ordering_to_string(Layout layout) {
    switch (layout) {
        case Layout::RowMajor: return "Row Major";
        case Layout::ColumnMajor: return "Column Major";
        default: return "Unknown Ordering";
    }
}

double performance_metrics(std::ofstream& fs, int M, int N, int K, float time_kernel, float time_cublas) {
    std::ostringstream oss;
    auto gflops = count_flops(M, N, K) / 1e9;

    oss << "Performance Metrics" << std::endl
        << "-------------------" << std::endl
        << "FLOP: " << gflops << " GFLOP" << std::endl
        << "Throughput: " << gflops / (time_kernel / 1e3) << " GFLOP/s" << std::endl
        << "Throughput cuBLAS: " << gflops / (time_cublas / 1e3) << " GFLOP/s" << std::endl
        << "Throughput (% of cuBLAS): " << time_cublas / time_kernel * 100.0 << "%" << std::endl
        << "Time (kernel): " <<  time_kernel << " ms" << std::endl
        << "Time (cuBLAS): " <<  time_cublas << " ms" << std::endl;

    std::cout << oss.str();
    fs << oss.str();

    return gflops;
}
