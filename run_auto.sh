#!/bin/bash   
#SBATCH --job-name=auto_reg%j.job
#SBATCH --output=auto_reg%j.out
#SBATCH -t 12:00:00
#SBATCH --mem 400G
#SBATCH -n 6
#SBATCH -c 2
module load python
python auto_regressive.py Who is
python auto_regressive.py Where are you
python auto_regressive.py When is
python auto_regressive.py Why are
python auto_regressive.py How come
python auto_regressive.py The girl
python auto_regressive.py My favorite
python auto_regressive.py I like
