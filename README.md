# Tensor Core Experiments

## Build and Run

### NVIDIA

```bash
# configure
cmake --preset nvidia-release

# build
cmake --build --preset nvidia-release --target example

# run directly the example
cmake --build --preset nvidia-release --target run_example

# format
cmake --build --preset nvidia-release --target format
```
