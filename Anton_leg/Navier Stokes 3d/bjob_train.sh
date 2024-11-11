#!/bin/sh
### General options
### -- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J train
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 01:00
# specify system resources  
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem = 4GB]"
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o batch_output/train_%J.out
#BSUB -e batch_output/train_%J.err
# -- end of LSF options --

cd ..
module load cuda/12.6.1  python3/3.10.13
source PINN/bin/activate
cd PINN_Bachelor/Anton_leg/Navier Stokes 3d/
python -u NavierStoketest.py
