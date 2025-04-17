cmake_minimum_required(VERSION 3.20)

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

# GPU architecture settings
if(AMD)
  if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES gfx90a)
  endif()
elseif(NVIDIA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89)
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

# Compiler flags
set(CMAKE_CXX_FLAGS "-Ofast -march=native -DNDEBUG -Wall")
# set(CMAKE_CUDA_FLAGS "-std=c++20 -O3 -DNDEBUG --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "-std=c++20 -O3 -DNDEBUG --use_fast_math --ftz=true --expt-relaxed-constexpr")
set(CMAKE_HIP_FLAGS "-Ofast -march=native -stdlib=libstdc++ -DNDEBUG -Wall")


find_package(OpenMP REQUIRED)

# CUDA setup
if(NVIDIA)
  execute_process(COMMAND spack location -i cuda OUTPUT_VARIABLE CUDA_PATH OUTPUT_STRIP_TRAILING_WHITESPACE)
  set(CUDAToolkit_ROOT "${CUDA_PATH}")
  set(CMAKE_PREFIX_PATH "${CUDAToolkit_ROOT};${CMAKE_PREFIX_PATH}")
  set(CMAKE_CUDA_COMPILER "${CUDAToolkit_ROOT}/bin/nvcc")
  find_package(CUDAToolkit REQUIRED)

  execute_process(COMMAND spack location -i cutensor OUTPUT_VARIABLE CUTENSOR_PATH OUTPUT_STRIP_TRAILING_WHITESPACE)
  set(CUTENSOR_INCLUDE_DIRS "${CUTENSOR_PATH}/include")
  set(CUTENSOR_LIBRARY_DIRS "${CUTENSOR_PATH}/lib/12")
  set(CUTENSOR_LIBRARIES "${CUTENSOR_LIBRARY_DIRS}/libcutensor.so")

  include_directories(${CUTENSOR_INCLUDE_DIRS})
  link_directories(${CUTENSOR_LIBRARY_DIRS})
endif()


# Source collection
# file(GLOB_RECURSE CPP_SOURCES "${CMAKE_SOURCE_DIR}/src/*.cpp")

file(GLOB_RECURSE ALL_CPP_SOURCES "${CMAKE_SOURCE_DIR}/src/*.cpp")

# Remove gemm.cpp from the list
list(FILTER ALL_CPP_SOURCES EXCLUDE REGEX ".*/gemm\\.cpp$")
set(CPP_SOURCES ${ALL_CPP_SOURCES})

add_library(gemm_with_openmp STATIC src/gemm.cpp)
target_link_libraries(gemm_with_openmp PUBLIC OpenMP::OpenMP_CXX)
target_include_directories(gemm_with_openmp PUBLIC ${CMAKE_SOURCE_DIR}/include)

if (NVIDIA)
  file(GLOB_RECURSE CU_SOURCES "${CMAKE_SOURCE_DIR}/src/kernels/cuda/*.cu" "${CMAKE_SOURCE_DIR}/src/gemm.cu" "${CMAKR_SOURCE_DIR}/src/cuda_utils.cpp")
endif()

# CUDA kernel object library
if(NVIDIA)
  add_library(cuda_kernels STATIC ${CU_SOURCES})
  set_target_properties(cuda_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
  )
#   target_compile_options(cuda_kernels PRIVATE
#     $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math --ftz=true -O3 -DNDEBUG --expt-relaxed-constexpr>
# )

target_compile_options(cuda_kernels PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 -DNDEBUG --expt-relaxed-constexpr>
)


  target_include_directories(cuda_kernels PRIVATE ${CMAKE_SOURCE_DIR}/include ${CONFIG_HEADER_DIR})
endif()


add_custom_target(format
    COMMAND astyle --style=google --suffix=none ${CPP_SOURCES} ${CU_SOURCES}
    COMMENT "Formatting source files with astyle"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)


# Executable and linkage
add_executable(${PROJECT_NAME}
  ${CPP_SOURCES}
)

target_include_directories(${PROJECT_NAME} PRIVATE include)
target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_BINARY_DIR})

if(NVIDIA)

  target_link_libraries(${PROJECT_NAME}
  PRIVATE
    gemm_with_openmp
    cuda_kernels
    CUDA::cublas
    ${PROJECT_BINARY_DIR}
    ${CUTENSOR_LIBRARIES}
)

# target_compile_definitions(${PROJECT_NAME} PRIVATE USE_CUDA=1)
elseif(AMD)
  target_link_libraries(${PROJECT_NAME}
    roc::rocthrust
    roc::hipsparse
  )
  # target_compile_definitions(${PROJECT_NAME} PRIVATE USE_HIP=1)
elseif(CPU)
  # target_compile_definitions(${PROJECT_NAME} PRIVATE CPU_ONLY=1)
endif()
