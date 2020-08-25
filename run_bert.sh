#!/bin/bash   
#SBATCH --job-name=usingbert%j.job
#SBATCH --output=usingbert%j.out
#SBATCH -t 12:00:00
#SBATCH --mem 400G
#SBATCH -n 6
#SBATCH -c 2
module load python
python usingbert.py Who
python usingbert.py Where
python usingbert.py When
python usingbert.py Why
python usingbert.py How
python usingbert.py The
python usingbert.py My
python usingbert.py I
