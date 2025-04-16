# cmake_minimum_required(VERSION 3.20)

# # Project definition
# set(PROJECT_NAME example)
# project(${PROJECT_NAME} C CXX CUDA)

# # Colored Output (Non-Windows)
# if(NOT WIN32)
#   string(ASCII 27 Esc)
#   set(ColourReset "${Esc}[m")
#   set(Green "${Esc}[32m")
# endif()

# # Build options
# option(CPU "Build for CPUs" OFF)
# option(NVIDIA "Build for NVIDIA GPUs" OFF)
# option(AMD "Build for AMD GPUs" OFF)

# if(CPU AND (NVIDIA OR AMD))
#   message(FATAL_ERROR "Cannot select both CPU and GPU builds.")
# endif()

# if(NVIDIA AND AMD)
#   message(FATAL_ERROR "Cannot select both NVIDIA and AMD GPU builds.")
# endif()

# # Configure GPU architectures
# if(AMD)
#   if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
#     set(CMAKE_HIP_ARCHITECTURES gfx90a)
#   endif()
#   enable_language(HIP)
# elseif(NVIDIA)
#   if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
#     set(CMAKE_CUDA_ARCHITECTURES 80 90)
#   endif()
#   enable_language(CUDA)
# endif()

# # Status messages
# if(CPU)
#   message(STATUS "${Green}Compiling for CPUs${ColourReset}")
# elseif(NVIDIA)
#   message(STATUS "${Green}Compiling for NVIDIA GPUs${ColourReset}")
#   add_definitions(-DNVIDIA)
# elseif(AMD)
#   message(STATUS "${Green}Compiling for AMD GPUs${ColourReset}")
#   add_definitions(-DAMD)
# else()
#   message(FATAL_ERROR "No target selected for the build.")
# endif()


# # C++ Standard and Build Flags
# set(CMAKE_CXX_STANDARD 20)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CXX_EXTENSIONS OFF)

# set(CMAKE_CXX_FLAGS "-Ofast -march=native -DNDEBUG -Wall")
# set(CMAKE_HIP_FLAGS "-Ofast -march=native -stdlib=libstdc++ -DNDEBUG -Wall")
# set(CMAKE_CUDA_FLAGS "-std=c++20 -O3 -DNDEBUG --expt-relaxed-constexpr")

# if(AMD)
#   find_package(hip REQUIRED)
#   find_package(rocprim REQUIRED CONFIG)
#   find_package(rocthrust REQUIRED CONFIG)
#   find_package(hipsparse REQUIRED CONFIG)

# endif()

# if (NVIDIA)

#   execute_process(
#     COMMAND spack location -i cuda
#     OUTPUT_VARIABLE CUDA_PATH
#     OUTPUT_STRIP_TRAILING_WHITESPACE
#   )

#   set(CUDAToolkit_ROOT "${CUDA_PATH}")
#   set(CMAKE_PREFIX_PATH "${CUDAToolkit_ROOT};${CMAKE_PREFIX_PATH}")

#   set(CUDAToolkit_INCLUDE_DIRS "${CUDAToolkit_ROOT}/include/")
#   set(CUDAToolkit_LIBRARY_DIR "${CUDAToolkit_ROOT}/lib64/")
#   set(CMAKE_CUDA_COMPILER "${CUDAToolkit_ROOT}/bin/nvcc")

#   find_package(CUDAToolkit REQUIRED)

#   message(STATUS "CUDA Toolkit: ${CUDAToolkit_INCLUDE_DIRS}")
#   message(STATUS "CUDA Libraries: ${CUDAToolkit_LIBRARY_DIR}")
#   message(STATUS "NVCC: ${CMAKE_CUDA_COMPILER}")

#   execute_process(
#     COMMAND spack location -i cutensor
#     OUTPUT_VARIABLE CUTENSOR_PATH
#     OUTPUT_STRIP_TRAILING_WHITESPACE
#   )

#   set(CUTENSOR_INCLUDE_DIRS "${CUTENSOR_PATH}/include")
#   set(CUTENSOR_LIBRARY_DIRS "${CUTENSOR_PATH}/lib/12")
#   set(CUTENSOR_LIBRARIES "${CUTENSOR_LIBRARY_DIRS}/libcutensor.so")

#   include_directories(${CUTENSOR_INCLUDE_DIRS})
#   link_directories(${CUTENSOR_LIBRARY_DIRS})
#   message(STATUS "cuTENSOR Include Directory: ${CUTENSOR_INCLUDE_DIRS}")
#   message(STATUS "cuTENSOR Library Directory: ${CUTENSOR_LIBRARY_DIRS}")
#   message(STATUS "cuTENSOR Library: ${CUTENSOR_LIBRARIES}")
# endif()


# file(GLOB_RECURSE CPP_SOURCES "${CMAKE_SOURCE_DIR}/src/*.cpp")


# set_source_files_properties(src/main.cpp PROPERTIES LANGUAGE CXX)

# if (NVIDIA)

#   add_executable(${PROJECT_NAME}
#     ${CPP_SOURCES}
#     $<TARGET_OBJECTS:cuda_kernels>
#   )
#   target_include_directories(${PROJECT_NAME} PRIVATE include)

#   file(GLOB_RECURSE GPU_SOURCE_FILES
#     "${CMAKE_SOURCE_DIR}/src/kernels/cuda/*.cu"
#   )

#   add_library(cuda_kernels OBJECT ${GPU_SOURCE_FILES})
#   set_target_properties(cuda_kernels PROPERTIES
#     LINKER_LANGUAGE CUDA
#   )
#   # Link the object files to the main executable
#   target_sources(${PROJECT_NAME} PRIVATE $<TARGET_OBJECTS:cuda_kernels>)

#   # Link CUDA libs
#   target_link_libraries(${PROJECT_NAME}
#     CUDA::cublas
#     ${CUTENSOR_LIBRARIES}
#   )

#   target_compile_definitions(${PROJECT_NAME} PRIVATE USE_CUDA=1)

#   # Ensure CUDA headers can be found
#   target_include_directories(cuda_kernels PRIVATE ${CMAKE_SOURCE_DIR}/include)

# elseif(AMD)
#   add_executable(${PROJECT_NAME} ${SOURCES})
#   set_source_files_properties(src/main.cpp PROPERTIES LANGUAGE HIP)
#   target_include_directories(${PROJECT_NAME} PRIVATE include)
#   target_link_libraries(${PROJECT_NAME}
#     roc::rocthrust
#     roc::hipsparse
#   )
#   target_compile_definitions(${PROJECT_NAME} PRIVATE USE_HIP=1)

# elseif(CPU)
#   add_executable(${PROJECT_NAME} ${SOURCES})
#   target_include_directories(${PROJECT_NAME} PRIVATE include)
#   target_compile_definitions(${PROJECT_NAME} PRIVATE CPU_ONLY=1)
# endif()


# add_custom_target(format
#     COMMAND astyle --style=google --suffix=none ${SOURCE_FILES}
#     COMMENT "Formatting source files with astyle"
#     WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
# )


# add_executable(${PROJECT_NAME} ${SOURCES})
# target_include_directories(${PROJECT_NAME} PRIVATE include)

# # GPU-specific configurations
# if(AMD)
#   target_link_libraries(${PROJECT_NAME} roc::rocthrust roc::hipsparse)
#   target_compile_definitions(${PROJECT_NAME} PRIVATE USE_HIP=1)
#   set_source_files_properties(src/main.cpp PROPERTIES LANGUAGE HIP)
# elseif(NVIDIA)
#   target_link_libraries(${PROJECT_NAME} ${CUTENSOR_LIBRARIES})
#   target_link_libraries(${PROJECT_NAME} CUDA::cublas)
#   target_compile_definitions(${PROJECT_NAME} PRIVATE USE_CUDA=1)
#   set_source_files_properties(src/main.cpp PROPERTIES LANGUAGE CXX)
# elseif(CPU)
#   target_compile_definitions(${PROJECT_NAME} PRIVATE CPU_ONLY=1)
# endif()

cmake_minimum_required(VERSION 3.20)
project(example LANGUAGES C CXX CUDA)

# Color messages
if(NOT WIN32)
  string(ASCII 27 Esc)
  set(ColourReset "${Esc}[m")
  set(Green "${Esc}[32m")
endif()

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
  enable_language(HIP)
elseif(NVIDIA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80 90)
  endif()
  enable_language(CUDA)
endif()

# Status
if(CPU)
  message(STATUS "${Green}Compiling for CPUs${ColourReset}")
elseif(NVIDIA)
  message(STATUS "${Green}Compiling for NVIDIA GPUs${ColourReset}")
  add_definitions(-DNVIDIA)
elseif(AMD)
  message(STATUS "${Green}Compiling for AMD GPUs${ColourReset}")
  add_definitions(-DAMD)
else()
  message(FATAL_ERROR "No target selected for the build.")
endif()

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
    CUDA_SEPARABLE_COMPILATION ON

  )
  target_compile_options(cuda_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math --ftz=true -O3 -DNDEBUG --expt-relaxed-constexpr>
)

  target_include_directories(cuda_kernels PRIVATE ${CMAKE_SOURCE_DIR}/include)
endif()

# Executable and linkage
add_executable(${PROJECT_NAME}
  ${CPP_SOURCES}
)

target_include_directories(${PROJECT_NAME} PRIVATE include)

if(NVIDIA)
  # target_link_libraries(${PROJECT_NAME}
  #   CUDA::cublas
  #   ${CUTENSOR_LIBRARIES}
  # )

  target_link_libraries(${PROJECT_NAME}
  PRIVATE
    gemm_with_openmp
    cuda_kernels
    CUDA::cublas
    ${CUTENSOR_LIBRARIES}
)

  target_compile_definitions(${PROJECT_NAME} PRIVATE USE_CUDA=1)
elseif(AMD)
  target_link_libraries(${PROJECT_NAME}
    roc::rocthrust
    roc::hipsparse
  )
  target_compile_definitions(${PROJECT_NAME} PRIVATE USE_HIP=1)
elseif(CPU)
  target_compile_definitions(${PROJECT_NAME} PRIVATE CPU_ONLY=1)
endif()
