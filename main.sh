#! /bin/bash
#SBATCH --partition=csedu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH -o ./logs/output.%j.out # STDOUT
#SBATCH --mail-user=sram@science.ru.nl
#SBATCH --time=144:00:00
#
~/radboud-naco-project/env/bin/python3 ~/radboud-naco-project/Sklearn-neat/main.py --generations=150 --population_size=50 --fitness_limit=0.9 --train_size=0.01 --test_size=0.005
