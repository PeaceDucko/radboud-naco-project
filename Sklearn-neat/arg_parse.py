import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--population_size", "-p",
                    help="Number of individuals in the population", default=4, type=int)
parser.add_argument("--generations", "-g",
                    help="Number of generations", default=10, type=int)
parser.add_argument("--population_limit", "-l",
                    help="Population limit percentage", default=0.9, type=float)

args = parser.parse_args()