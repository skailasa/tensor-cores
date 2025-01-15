#!/bin/bash

# Collect Spack paths
CUDA_PATH=$(spack location -i cuda)
CUTENSOR_PATH=$(spack location -i cutensor)

# Generate JSON configuration
cat <<EOL > ./c_cpp_properties.json
{
    "configurations": [
        {
            "name": "Lumi",
            "includePath": [
                "\${workspaceFolder}/include/",
                "/opt/rocm/include/",
                "/usr/include",
                "/usr/local/include",
                "/usr/include/c++/13/",
                "/usr/include/c++/13/backward",
                "${env:ROCM_PATH}/include",
            ],
            "defines": [
                "___HIP_PLATFORM_AMD__",
                "AMD"
            ],
            "compilerPath": "/opt/rocm/bin/hipcc",
            "cppStandard": "c++20",
            "intelliSenseMode": "linux-gcc-x86"
        },
        {
            "name": "NVidia Workstation",
            "includePath": [
                "\${workspaceFolder}/include/",
                "/usr/include",
                "/usr/local/include",
                "/usr/include/c++/13/",
                "/usr/include/x86_64-linux-gnu/c++/13/",
                "/usr/include/c++/13/backward",
                "/usr/lib/gcc/x86_64-linux-gnu/13/include",
                "/usr/include/x86_64-linux-gnu",
                "${CUDA_PATH}/include/",
                "${CUTENSOR_PATH}/include/"
            ],
            "defines": [
                "__CUDAACC__",
                "NVIDIA"
            ],
            "compilerPath": "${CUDA_PATH}/bin/nvcc",
            "cppStandard": "c++20",
            "intelliSenseMode": "linux-gcc-x86"
        }
    ],
    "version": 4
}
EOL

echo "Updated .vscode/c_cpp_properties.json with Spack paths."
