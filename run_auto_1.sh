#!/bin/bash   
#SBATCH --job-name=auto_reg%j.job
#SBATCH --output=auto_reg%j.out
#SBATCH -t 12:00:00
#SBATCH --mem 400G
#SBATCH -n 6
#SBATCH -c 2
module load python
python auto_regressive.py Who
python auto_regressive.py Where
python auto_regressive.py When
python auto_regressive.py Why
python auto_regressive.py How
python auto_regressive.py The
python auto_regressive.py My
python auto_regressive.py I
