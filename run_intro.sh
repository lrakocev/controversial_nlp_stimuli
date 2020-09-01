#!/bin/bash   
#SBATCH --job-name=auto_reg%j.job
#SBATCH --output=auto_reg%j.out
#SBATCH -t 12:00:00
#SBATCH --mem 400G
#SBATCH -n 6
#SBATCH -c 2
module load python
python introtransformers.py Who
python introtransformers.py Where
python introtransformers.py When
python introtransformers.py Why
python introtransformers.py How
python introtransformers.py The
python introtransformers.py My
python introtransformers.py I
