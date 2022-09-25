#!/bin/bash
#
#SBATCH --array=0,1,2,3,4,5
#SBATCH --job-name=DoubleQ_ep10.py
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G 
#SBATCH --time=2:20:00
#SBATCH --account=rrg-cbravo
#SBATCH --output=DoubleQ_ep10.%A%_a.out


module load python scipy-stack
source ~/Tutorial2/bin/activate
python  /home/salfonso/projects/def-cbravo/salfonso/RL_GitHub/Double_Qlearning/DQ_learning_epsilon0.1.py $SLURM_ARRAY_TASK_ID