# Tensor Core Experiments

## Contents


### `example`

#### Compile

**LUMI**

```bash

# Required for build on login node
module load LUMI/23.09
module load partition/G
module load rocm
module load gcc
spack env activate hypre-rocm
spack load cmake

# Build example
cd example
mkdir build && cd build
cmake -Damd=ON ..
make
```

### `include`

Contains utilities


## LUMI

- MI250x, 2 GCD per unit.
- Each GCD has 110 usable CUs
- All CUs share L2 - each of which is 8MD, and divided in 32 slices capaable of delivering 128B/clock/slice. 6.96TB/s of peak bandwidth
- Can perform atomics on data in L2, can use to coordinate commmunication across GPU

