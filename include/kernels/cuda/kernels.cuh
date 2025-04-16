#include <cuda_runtime.h>
#include "1_naive.cuh"
#include "2_shared_mem_cache_blocking.cuh"
#include "3_1d_blocktiling.cuh"
#include "4_2d_blocktiling.cuh"
#include "5_vectorise_smem.cuh"