#include "cuda_utils.hpp"
#include <cstdio>
#include <cuda_runtime.h>
#include <fstream>
#include <iomanip>
#include <iostream>

std::string cache_config_to_string(cudaFuncCache config) {
  switch (config) {
  case cudaFuncCachePreferNone:
    return "PreferNone";
  case cudaFuncCachePreferShared:
    return "PreferShared";
  case cudaFuncCachePreferL1:
    return "PreferL1";
  case cudaFuncCachePreferEqual:
    return "PreferEqual";
  default:
    return "Unknown Cache Config";
  }
}

void device_info(std::ofstream &fs) {
  int device_id;
  cudaGetDevice(&device_id);

  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, device_id);

  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2);

  int warps_per_sm = props.maxThreadsPerMultiProcessor / props.warpSize;

  oss << "Device Info\n"
      << "===========\n"
      << "Device ID                        : " << device_id << "\n"
      << "Name                             : " << props.name << "\n"
      << "Compute Capability               : " << props.major << "."
      << props.minor << "\n"
      << "Total Global Memory              : "
      << props.totalGlobalMem / (1024 * 1024) << " MB\n"
      << "Memory Bus Width                 : " << props.memoryBusWidth
      << " bits\n"
      << "Multiprocessor Count (SMs)       : " << props.multiProcessorCount
      << "\n"
      << "Max Threads per Block            : " << props.maxThreadsPerBlock
      << "\n"
      << "Max Threads per Multiprocessor   : "
      << props.maxThreadsPerMultiProcessor << "\n"
      << "Threads per Warp                 : " << props.warpSize << "\n"
      << "Estimated Warps per SM           : " << warps_per_sm << "\n"
      << "Max Registers per Block          : " << props.regsPerBlock << "\n"
      << "Max Shared Mem per Block         : " << props.sharedMemPerBlock / 1024
      << " KB\n"
      << "Shared Mem per Multiprocessor    : "
      << props.sharedMemPerMultiprocessor / 1024 << " KB\n"
      << "Constant Memory                  : " << props.totalConstMem / 1024
      << " KB\n"
      << "---------------------------------------------\n";

  std::cout << oss.str();
  fs << oss.str();
}
