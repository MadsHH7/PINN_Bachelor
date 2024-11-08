#!/bin/sh
### General options
### -- specify queue --
#BSUB -q gpuv100

### gpua100 or gpuv100

### -- set the job Name --
#BSUB -J PINN_Test_with_Data

### -- ask for number of cores (default: 1) --
#BSUB -n 4

### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"

### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 11:00
# specify system resources
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"

### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --

###BSUB -o /zhome/e1/d/168534/Desktop/Test_%J.out
#BSUB -o batch_outputs/Test_%J.out
#BSUB -e batch_outputs/Test_%J.err
# -- end of LSF options --

### module load python3/3.10.13 cuda/12.6.1 cudnn/v8.9.7.29-prod-cuda-12.X cudnn/v8.9.7.29-prod-cuda-12.X
module load cuda/12.6.1 python3/3.10.13
source .venv/bin/activate
cd /zhome/e1/d/168534/Desktop/Bachelor_PINN/PINN_Bachelor/batch_outputs/
python -u 3D_Pipe.py

### Calls with bsub < [Name of sh script]
### Check status' bjobs
### showstart #id
### Kill jobs with bkill #id