#!/bin/bash

# Copy/paste this job script into a text file and submit with the command:
#    sbatch thefilename

#SBATCH --time=24:00:00   # walltime limit (HH:MM:SS)
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=8   # 8 processor core(s) per node 
#SBATCH --mem=10G   # maximum memory per node
#SBATCH --gpus=4
#SBATCH --partition=gpu    # gpu node(s)
#SBATCH --job-name="YING"
#SBATCH --mail-user=swyoon@kias.re.kr   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output="OUTPUT_%j.out" # job standard output file (%j replaced by job id)
#SBATCH --error="ERROR_%j.out" # job standard error file (%j replaced by job id)

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate gcd

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8106 finetune.py --name cifar10 --img_shape 32
