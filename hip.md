# AMD

## Register Pressure in AMD GPUs

A high number of concurrent wavefronts runnning on the same CU enables the GPU to hide the time spent in accessing global memory, which is higher than the time needed to perform a compute operation.

The term occupancy represents the maximum number of wavefronts that can be potentially run on the same CU at the same time. In general, having higher occupancy helps achieve better performance by hiding costly memory accesses and other operations.


In CDNA2 architectures.

- The vector general purpose registers (VGPRs) are used to store data that is not uniform across the wavefront, that is, data that is different for each work item in the wavefront. The most general purpose registers available in the CU and are directly manipulated by the vector ALU (VALU).
- The VALU is responsible for executing most of the work in the CU, including FLOPS, loads etc.

- The Scalar General Purpose Registers, SGPRs a set of registers used to store data that is known to be uniform at compile time. Manipulated by the Scalar ALU. Can only be used for a limited set of operations, like integer and logical.

- The local data share (LDS) is a fast on-CU software managed memory, that can be used to efficiently share data between all work items on a block.

Occupancy is limited by the hardware design and resource limitations dictated by the kernel. For example, each CU of the AMD CDNA2 based GPU has four sets of wavefront buffers. One per Execution Unit (also called SIMD unit), and four EUs per CU. Each EU can manage at most eight wavefronts. This means that the physical limit to occupancy is at most 32 wavefronts per CU. Need to look up hardware specifics on how many VGPRs this corresponds to per-wavefront.


### Register Spilling

When the number of registers requested becomes too high, performance is penalised by register pressure which leads to low occupancy and scratch memory usage.

Sometimes the compiler may decide that its fruitful to reach a better level of occupancy even though the request for registers is higher than the limit for a given configuration. This higher level of occupancy can be achieved by saving some variables in scratch memory. A portion of local memory, private to a thread. Backed by global memory, but much slower than register memory. This is called register spilling.

Achieving higher occupancy by saving a few registers could on ocassion provide a substantial performance benefit as opposed to a lower occupancy, without any scratch memory usage.

### Reducing Register Pressure

The compiler applies heuristic techniques to maximise occupancy by minimising the number of registers needed by certain kernels. These heuristic techniques can sometimes fail to be optimal - doing this properly is known to be NP Hard.

The number of registers used by GPU kernels can be detected in two ways

1. Compile the file containing the kernels with `-Rpass-analyze=kernel-resource-usage` flag. Prints to the screeen the resource usage of each kernel at compile time.
2. Compile with the flag `--save-temps` and look in `hip-amdgcp-amd-amdhsa-gfx90a.s` for the `.vgpr_spill_count`

How to reduce if confirmed?

- Set the `__launch_bounds__` qualifier for each kernel. By default, the compiler assumes that the block size of each kernel is 1024 work items. When this parameter is defined, the compiler can allocate registers appropriately, thus potentially lowering the register pressure.
- Move variable definition/assignment close to where they are used. Defining one or multiple variables at the top of a GPU kernel, and using them at the bottom forces the compiler to keep those variables in a register until they are used. Thus impacting the possibility of using those registers for more performance critical variables.
- Avoid alocating data on the stack, it lives in scratch memory by default and may be stored into registers by the compiler as an optimisation step.
- Avoid passing big objects as kernel arguments, function arguments re stack allocated and may be saved into registers as an optimisation. Storing these in `constant` if needed at all may help.
- Avoid writing large kernels, with many function calls - including math functions and assertions. The compiler always inlines device functions, including math functions and assertions. Having many of these function calls introduces extra code and potentially higher register pressure.
- keep loop unrolling under control. Unrolling increases register pressure as more variables need to be stored in registers at the same time. Note that clang tends to favour loop unrolling more literally than other compilers.
- Manually spill to LDS memory. As a last resort, can be beneficial to use some LDS memory to manually store variables and save a few registers per thread.



## HIP