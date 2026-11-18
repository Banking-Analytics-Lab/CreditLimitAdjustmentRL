#!/bin/bash
#
#SBATCH --array=0,1,2,3,4,5
#SBATCH --job-name=DoubleQ_ep10.py
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G 
#SBATCH --time=2:20:00
#SBATCH --account=<your-allocation>
#SBATCH --output=DoubleQ_ep10.%A%_a.out


module load python scipy-stack
source "${VENV_DIR:-venv}/bin/activate"
python  "${CODE_DIR:-.}/DQ_learning_epsilon0.1.py" $SLURM_ARRAY_TASK_ID