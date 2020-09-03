#!/bin/bash
#SBATCH --job-name=judge%j.job
#SBATCH --output=judge%j.out
#SBATCH -t 12:00:00
#SBATCH --mem 400G
#SBATCH -n 6
#SBATCH -c 2
module load python
python judge_sentences.py
