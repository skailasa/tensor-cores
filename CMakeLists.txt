cmake_minimum_required(VERSION 3.20)

# Project definition
set(PROJECT_NAME example)
project(${PROJECT_NAME} C CXX)

# Colored Output (Non-Windows)
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

# Configure GPU architectures
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

# Status messages
if(CPU)
  message(STATUS "${Green}Compiling for CPUs${ColourReset}")
elseif(NVIDIA)
  message(STATUS "${Green}Compiling for NVIDIA GPUs${ColourReset}")
elseif(AMD)
  message(STATUS "${Green}Compiling for AMD GPUs${ColourReset}")
else()
  message(FATAL_ERROR "No target selected for the build.")
endif()

# Add formatting target
add_custom_target(
  format
  COMMAND astyle --options=${CMAKE_SOURCE_DIR}/.astylerc --recursive ${CMAKE_SOURCE_DIR}/*.cpp ${CMAKE_SOURCE_DIR}/*.h
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  COMMENT "Formatting code with Astyle"
)

# C++ Standard and Build Flags
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS "-Ofast -march=native -DNDEBUG -Wall")
set(CMAKE_HIP_FLAGS "-Ofast -march=native -stdlib=libstdc++ -DNDEBUG -Wall")
set(CMAKE_CUDA_FLAGS "-std=c++20 -O3 -DNDEBUG --expt-relaxed-constexpr")

# Dependencies
find_package(MPI REQUIRED)
find_package(Boost 1.70 REQUIRED program_options)

if(AMD)
  find_package(hip REQUIRED)
  find_package(rocprim REQUIRED CONFIG)
  find_package(rocthrust REQUIRED CONFIG)
  find_package(hipsparse REQUIRED CONFIG)
endif()

# Source and executable
set(SOURCES
  example/main.cpp
)

add_executable(${PROJECT_NAME} ${SOURCES})
target_include_directories(${PROJECT_NAME} PRIVATE include)

# GPU-specific configurations
if(AMD)
  target_link_libraries(${PROJECT_NAME} roc::rocthrust roc::hipsparse)
  target_compile_definitions(${PROJECT_NAME} PRIVATE USE_HIP=1)
  set_source_files_properties(example/main.cpp PROPERTIES LANGUAGE HIP)
elseif(NVIDIA)
  target_compile_definitions(${PROJECT_NAME} PRIVATE USE_CUDA=1)
  set_source_files_properties(example/main.cpp PROPERTIES LANGUAGE CUDA)
elseif(CPU)
  target_compile_definitions(${PROJECT_NAME} PRIVATE CPU_ONLY=1)
endif()
