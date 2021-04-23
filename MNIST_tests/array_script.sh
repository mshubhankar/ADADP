#!/bin/bash
#SBATCH --array=0-8
#SBATCH --gres=gpu:1         # Request GPU "generic resources"
#SBATCH --cpus-per-task=2       # Cores proportional to GPUs: 6 on Cedar, 10 on Béluga, 16 on Graham.
#SBATCH --mem=5000M             # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=0-1:00:00          # DD-HH:MM:SS
module load python/3.7 cuda cudnn
SOURCEDIR=/home/s3mohapa/projects/rrg-xihe/s3mohapa/adadp
source $SOURCEDIR/env/bin/activate
python $SOURCEDIR/main_adadp.py $SOURCEDIR/data $SLURM_ARRAY_TASK_ID