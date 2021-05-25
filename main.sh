#! /bin/bash
#SBATCH --partition=csedu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<username>@science.ru.nl
#SBATCH --time=12:00:00
#
~/radboud-naco-project/env/bin/python3 ~/radboud-naco-project/Sklearn-neat/main.py
