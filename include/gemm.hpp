#pragma once

#include <types.hpp>

void runSgemmCpu(Layout layout, int M, int N, int K, float alpha, float *A,
                 float *B, float beta, float *C);