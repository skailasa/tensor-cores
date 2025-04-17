cmake_minimum_required(VERSION 3.20)

# Function to locate spack packages on system
function(find_spack_package name out_path)
  execute_process(COMMAND spack location -i ${name} OUTPUT_VARIABLE _path OUTPUT_STRIP_TRAILING_WHITESPACE)
  set(${out_path} "${_path}" PARENT_SCOPE)
endfunction()


# Build options
option(CPU "Build for CPUs" OFF)
option(NVIDIA "Build for NVIDIA GPUs" OFF)
option(AMD "Build for AMD GPUs" OFF)

if(CPU AND (NVIDIA OR AMD))
  message(FATAL_ERROR "Cannot select both CPU and GPU builds.")
endif()
if(NVIDIA AND AMD)
  message(FATAL_ERROR "Cannot select both NVIDIA and AMD GPU builds.")
endif()

# GPU architecture settings, must come before enable language
if(AMD)
  if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES gfx90a)
  endif()
elseif(NVIDIA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80 90)
  endif()
endif()

if(AMD)
  enable_language(HIP)
elseif(NVIDIA)
  enable_language(CUDA)
endif()

# Status
if(CPU)
message(STATUS "🚀 Building for CPU")
elseif(NVIDIA)
  message(STATUS "🚀 Building for NVIDIA GPUs (CUDA)")
elseif(AMD)
  message(STATUS "🚀 Building for AMD GPUs (HIP)")
else()
  message(FATAL_ERROR "❌ No target selected for build")
endif()

project(example LANGUAGES C CXX)

# configure a header file to document various build config options
configure_file(${CMAKE_SOURCE_DIR}/Config.h.in ${CMAKE_BINARY_DIR}/Config.h)
set(CONFIG_HEADER_DIR ${PROJECT_BINARY_DIR})

# Standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Required system libraries
find_package(OpenMP REQUIRED)

# CUDA setup
if(NVIDIA)
  find_spack_package(cuda CUDA_PATH)
  set(CUDAToolkit_ROOT "${CUDA_PATH}")
  set(CMAKE_PREFIX_PATH "${CUDAToolkit_ROOT};${CMAKE_PREFIX_PATH}")
  set(CMAKE_CUDA_COMPILER "${CUDAToolkit_ROOT}/bin/nvcc")
  find_package(CUDAToolkit REQUIRED)

  find_spack_package(cutensor CUTENSOR_PATH)

  file(GLOB CUTENSOR_LIBRARY "${CUTENSOR_PATH}/lib/*/libcutensor.so")
  if(NOT CUTENSOR_LIBRARY)
    file(GLOB CUTENSOR_LIBRARY "${CUTENSOR_PATH}/lib/libcutensor.so")
  endif()

  # Make sure that only one file is matched by extracting first hit
  list(GET CUTENSOR_LIBRARY 0 CUTENSOR_LIBRARY)
  set(CUTENSOR_INCLUDE_DIR "${CUTENSOR_PATH}/include")

  add_library(CUTENSOR::cutensor SHARED IMPORTED)
  set_target_properties(CUTENSOR::cutensor PROPERTIES
    IMPORTED_LOCATION "${CUTENSOR_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${CUTENSOR_INCLUDE_DIR}"
  )

endif()

# Specify sources
set(CPP_SOURCES
  ${CMAKE_SOURCE_DIR}/src/main.cpp
  ${CMAKE_SOURCE_DIR}/src/utils.cpp
)

if (NVIDIA)
 list(APPEND CPP_SOURCES ${CMAKE_SOURCE_DIR}/src/cuda_utils.cpp)
endif()

# Add a target for computing GEMM on CPU
add_library(gemm_with_openmp STATIC ${CMAKE_SOURCE_DIR}/src/gemm.cpp)
target_link_libraries(gemm_with_openmp PUBLIC OpenMP::OpenMP_CXX)
target_include_directories(gemm_with_openmp PUBLIC ${CMAKE_SOURCE_DIR}/include)

# Add a target for all the CUDA kernels
function(configure_cuda_kernels OUT_VAR)
  add_subdirectory(src/kernels/cuda)

  set(local_sources
    "${CMAKE_SOURCE_DIR}/src/gemm.cu"
    "${CMAKE_SOURCE_DIR}/src/cuda_utils.cpp"
  )

  list(APPEND local_sources ${CU_SOURCES})
  set(${OUT_VAR} "${local_sources}" PARENT_SCOPE)
endfunction()

if (NVIDIA)
  configure_cuda_kernels(CU_SOURCES)
  add_library(cuda_kernels STATIC ${CU_SOURCES})

  target_compile_options(cuda_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math --ftz=true -O3 -DNDEBUG --expt-relaxed-constexpr>
  )

  target_link_libraries(cuda_kernels PRIVATE CUDA::cublas CUTENSOR::cutensor)

  target_include_directories(cuda_kernels PRIVATE
    ${CMAKE_SOURCE_DIR}/include/
    ${CONFIG_HEADER_DIR}
  )
endif()

# Add a target for the example code
add_executable(${PROJECT_NAME}
  ${CPP_SOURCES}
)

target_include_directories(${PROJECT_NAME}
  PRIVATE
    ${CMAKE_SOURCE_DIR}/include
  PUBLIC
    ${PROJECT_BINARY_DIR}  # so consumers can see Config.h if needed
)

if(NVIDIA)
  target_link_libraries(${PROJECT_NAME}
  PRIVATE
    gemm_with_openmp
    cuda_kernels
)

elseif(AMD)
  # target_link_libraries(${PROJECT_NAME}
  #   roc::rocthrust
  #   roc::hipsparse
  # )
elseif(CPU)
endif()

# Custom commands
add_custom_target(format
    COMMAND astyle --style=google --suffix=none ${CPP_SOURCES} ${CU_SOURCES}
    COMMENT "Formatting source files with astyle"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)