#!/bin/bash

module load LUMI/23.09
module load partition/G
module load rocm
module load gcc
spack env activate lumi
spack load cmake