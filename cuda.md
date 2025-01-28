# CUDA

Notes on CUDA, especially related to Tensor Cores. But general revision material too.

## Contents

- [Programming Model](#programming-model)
- [Programming Interface](#programming-interace)

- [Architecture Notes](#architecture-notes)


## Programming Model

- Limit to the number of threads per block, since all threads of a block are expected to reside in a single SM.
    - On current architectures a thread block may contain up to **1024 threads at most**.

- Thread blocks are required to execute independently, i.e. we should be able to schedule them in any order across any number of cores.

#### Logical Memory Hierarchy

- Each thread has a private local memory
- Each thread block has shared memory visible to all owned threads, same lifetime as block
- Two read only memory spaces accessible by all threads
    - constant and texture memory spaces
- Global constant and texture memory are persistent across kernel launches by same application

## Programming Interace

- The CUDA runtime probides C/C++ functions that execute on the host to allocate and deallocate device memory, transfer data and manage multiple devices etc.

- Built on top of lwe level C API, and CUDA driver API, also accessible programmitically.

- Kernels can be written using the CUDA ISA called PTX

- nvcc is compiler driver that simplifies the process of compiling C++ or PTX code (assembly for GPU) - binary form (CUBIN).

- Any PtX code loaded by an application at runtime is compiled furthe rto binary code by the device driver (JIT).


### CUDA Runtime

- Runtime implemented in `cudart` library, linked to the application.
- `cudaInitDevice()` and `cudaSetDevice()` initalise the runtime and primary context associated with a specified device. Otherwise will automatically use whatever device 0 is set as on multi-gpu devices.
- The runtime creates a CUDA context for each device in the system. Shared among all host threads of an application.
- Device memroy allocated either as linear memory or as CUDA arrays. CUDA array are opaque memory layouts, optimised for texture fetching.

- Linear memory is allocated in a single unified address space, which means that it separately allocated entities can reference one another via pointers. E.G trees/linked lists.

- Linear memory allocated using `cudaMalloc` and `cudaFree`, data transfer accomplished with `cudaMemcpy`.

- Can also allocate with `cudaMallocPitch()` and `cudaMalloc3D()`, recommended for allocations of 2D and 3D arrays to ensure that allocation is properly padded to meet alignement requirements and ensure that best performance when accessing row addresses or performing copies between 2D arrays and other regions of device memory. Corresponding `cudaMemcpy2D()`, `cudaMemcpy3d()`.

#### Device Memory L2 Access Management

- When a kernel accesses a data region in global memory repeatedly, accesses can be considered to be persisting. On the other hand if data is only accessed once can be considered to be streaming.

- Devices capabale of CC >8.0 can influence what resides in L2

#### Shared Memory

- allocated with `___shared__` space specifier. Refers to on chip (on SM) memory. local to each SM.

#### Page-Locked Host Memory

- pinned host memory, as opposed to regular pageable host memory allocated with `malloc()` i.e. cannot be paged out to disk via virtual memory addressing.

- local memory is local *logically* to each thread. Variables stored in local memory are private to a thread, and no other thread can access them.
- Local memory is allocated in a GPU's DRAM. Each thread has its own portion of local memory.
- Local memory is automatically used by the CUDA compiler when a thread's register usage exceeds the hardware limit.
- When a variable is declared with the local attribute or its size/dynamic nature makes it unsuitable for registers - large arrays or unknown compile time sizes.
- Any variable if the kernel uses more registers than available - known as **register spilling**.

- Can examine PTX for register spilling. Subsequent architecture specific compilation phases may mean cubin code has to be examine., `lmem` shows local memory usage per kernel.

- Local memory space resides in device memory, so access has same high latency as global memory accesses and are subjec tto same requirements for memory coalescing.

- However, localmemory is organised such that consecutive 32-bit words are accessed by consecutive thread IDs. Accesses are therefore fully coalesced as long as all threads in a warp access the same relative address. e.g. same index in a array variable.


#### Shared Memory

On chip - local to each SM.

- divided into equally sized memory modules called banks, which can be accessed simulataneously, therefore overall bandwidth is _n_ times as high as the bandwidth of a single module.

- However, if two addresses of a memory request fall in the same memory bank, there is a bank conflict and the access has to be serialised. The hardware splits a memory request with bank conflicts into as many separate conflic-free requests as necessary.

#### Constant Memory

- resides in device memory and is cached in the constant cache.

#### Texture and Surface Memory

- Reside in device memory and are cached in texture cache.
- optimised for 2D spatial locality, so that threads of the same warp that read texture or surface addresses that are close together in 2D will achieve the best performance.


## Architecture Notes

- [ADA Generation](#ada-generation-architecture)
- [Hopper](#hopper)
- [SIMT Architecture](#simt-architecture)

### General NVIdia Architecture reference

### SIMT Architecture

- The GPU manages, schedules and executes threads in groups of 32 parallel threads called **warps**.
- Individual threads composing a warp start together at the same program address, but have their own instruction address counter an dregister state and tare therefore free to branch and execute independently.
- when a multiptrocessor is given one or more thread block to execute, it partitions them into warps and each warp gets scheduled by a warp scheduler for exectuion.
- Blocks are partitioned into warps in the same way each time. Each warp contains threads of consecutive, increasing thread IDs, with the first warp containing thread 0.

- A warp executes one common instruction a time. So full efficiency is realised when all 32 threads of a warp agree on their execution path;
- If threads of a warp diverge via branching, the warp executes eahc branch path taken, disabling threads (idle) which are not on that branch.
- Branch divergence only happens within a warp, different warps execute independently regardless of whether they are executing common or disjoint code paths.


- The execution context (PC, registers etc) for each warp processed by a GPU is maintained on-chip during the entire lifetime of the warp. Therefore switching contexts has no cost, at every instruction issue time the warp scheduler selects a warp that has threads ready to go to choose its next isntruction.

### Ada Generation Architecture

### Occupancy
- Max number of concurrent warps per SM is 48
- Register file size is 64K 32-bit registers per SM
- Max number of registers per thread is 255
- Max number of thread blocks per SM is 24
- Max shared memory per SM is 100KB
- Max shared memory per thread block is 99KB

Note
- Must compile explicitly for CC 8.9 to benefit from increased SP throughput, increased by 2X since CC 8.0

- Increased L2 Cache to 98304 KB

CUDA reserves 1KB of shared memory per thread block. Hence GPUs with CC 8.9 can address up to 99KB of shared memory in a single thread block.


### Hopper


## Performance Guidelines

- Maximise parallel execution to achieve max utilisation
- Optimise memory usage to achieve max throughput
- Optimise instructin usage to achieve max instruction throughput
- Minimise memory thrashing

- Note can execute multiple kernels concurrently on a device.

- At a lowe rlevel the application should maximise parallel execution between the various funcional units within a GPU
- Utilisation is therefore directly linked to the number of resident warps.
