#!/bin/bash


# USER@home.ccs.ornl.gov
#

# Data transfer
#scp USER09@dtn.ccs.ornl.gov:/ccs/home/USER/torchlr/examples/file .


ml python/3.13.0
#python3 -m venv .venv
source .venv/bin/activate

module load PrgEnv-gnu/8.7.0
module load cpe/26.03
module load miniforge3/23.11.0-0
module load rocm/7.1.1
module load craype-accel-amd-gfx90a

# Because using a non-default CPE
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH

pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/rocm7.1

