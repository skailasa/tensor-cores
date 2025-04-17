#include "Config.h"

#ifdef NVIDIA
#include "kernels/cuda/kernels.cuh"
#endif

#ifdef AMD
#include "kernels/hip/kernels.cuh"
#endif