#! /bin/bash
#SBATCH --partition=csedu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH -o ./logs/output.%j.out # STDOUT
#SBATCH --mail-user=sram@science.ru.nl
#SBATCH --time=01:00:00
#
~/radboud-naco-project/env/bin/python3 ~/radboud-naco-project/Sklearn-neat/main.py --generations=1 --population_size=10 --population_limit=0.1
