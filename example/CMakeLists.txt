cmake_minimum_required(VERSION 3.20)

# Colored Output (Non-Windows)
if(NOT WIN32)
  string(ASCII 27 Esc)
  set(ColourReset "${Esc}[m")
  set(Green "${Esc}[32m")
endif()

# Build options
option(cpu "Build for CPUs" OFF)
option(nvidia "Build for NVIDIA GPUs" OFF)
option(amd "Build for AMD GPUs" OFF)

if(cpu AND (nvidia OR amd))
  message(FATAL_ERROR "Cannot select both CPU and GPU builds.")
endif()

if(nvidia AND amd)
  message(FATAL_ERROR "Cannot select both NVIDIA and AMD GPU builds.")
endif()

# Project definition
set(PROJECT_NAME example)

if(amd)
  if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES gfx90a)
  endif()
  project(${PROJECT_NAME} C CXX HIP)
elseif(nvidia)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80 90)
  endif()
  project(${PROJECT_NAME} C CXX CUDA)
else()
  project(${PROJECT_NAME} C CXX)
endif()

# Status message
if(cpu)
  message(STATUS "${Green}Compiling for CPUs${ColourReset}")
elseif(nvidia)
  message(STATUS "${Green}Compiling for NVIDIA GPUs${ColourReset}")
elseif(amd)
  message(STATUS "${Green}Compiling for AMD GPUs${ColourReset}")
else()
  message(FATAL_ERROR "No target selected for the build.")
endif()

# C++ Standard and Build Flags
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-Ofast -march=native -DNDEBUG -Wall")
set(CMAKE_HIP_FLAGS "-Ofast -march=native -stdlib=libstdc++ -DNDEBUG -Wall")
set(CMAKE_CUDA_FLAGS "-std=c++20 -O3 -DNDEBUG --expt-relaxed-constexpr")

# Dependencies
find_package(MPI REQUIRED)
find_package(Boost 1.70 REQUIRED program_options)

if(amd)
  find_package(hip REQUIRED)
  find_package(rocprim REQUIRED CONFIG)
  find_package(rocthrust REQUIRED CONFIG)
  find_package(hipsparse REQUIRED CONFIG)
endif()

# Source and executable
add_executable(${PROJECT_NAME} main.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE "../")

if(amd)
  target_link_libraries(${PROJECT_NAME} roc::rocthrust roc::hipsparse)
  add_definitions(-DUSE_HIP=1)
  set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)
elseif(nvidia)
  add_definitions(-DUSE_CUDA=1)
  set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)
elseif(cpu)
  add_definitions(-DCPU_ONLY=1)
endif()
