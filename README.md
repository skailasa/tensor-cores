# Tensor Core Experiments

Assume column major inputs, to match those expected by cuBLAS.

## Contents

-  `example` - ...
- `foo` - ...

#### **NVidia Workstation**

RTX 6000 Ada Generation Data Sheet

- Ada Lovelace Architecture AD102
- 18,176 CUDA cores
- 568 Tensor Cores
- 2.5 GHz
- FP32 91.1 TFLOP/s
- TF32 ~ 362.6 TFLOP/s (2x with sparsity)
- FP16/BF16 ~ 725.3 TFLOP/s (with sparsity 2:4 structured)

-  The "with sparsity" numbers refer to 2:4 structured sparsity, where 50% of operands are zero and optimized by hardware.
- If you're not explicitly using sparse data + sparse kernels, you're hitting ~362 TFLOP/s for TF32 and ~362 TFLOP/s for FP16/BF16.

Note the data sheet reports performance for all cores/tensor cores running every cycle with no bank conflicts, shared memory stalls or launch overheads.



#### *LUMI**

Install spack environment

```bash
spack env create lumi spack/lumi.yaml && spack env activate lumi
spack install
```

```bash
# Required for build on login node
module load LUMI/23.09
module load partition/G
module load rocm
module load gcc
spack env activate lumi
spack load cmake

# Build binaries
mkdir build && cd build
cmake -DAMD=ON ..
make
```

# Run example
```bash
srun --ntasks=1 ./example
```

### `include`

Contains utilities


## LUMI

- MI250x, 2 GCD per unit.
- Each GCD has 110 usable CUs
- All CUs share L2 - each of which is 8MD, and divided in 32 slices capaable of delivering 128B/clock/slice. 6.96TB/s of peak bandwidth
- Can perform atomics on data in L2, can use to coordinate commmunication across GPU


Request GPU node

```bash
salloc --nodes=1 --ntasks-per-node=8 --gpus-per-node=8 --time=01:0:00 --partition=dev-g --account=$ACCOUNT
```

### Lint

```bash
astyle --options=.astylerc *.cpp *.hpp --recursive
```
