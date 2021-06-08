import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--population_size", "-p",
                    help="Number of individuals in the population", default=4, type=int)
parser.add_argument("--generations", "-g",
                    help="Number of generations", default=10, type=int)
parser.add_argument("--population_limit", "-l",
                    help="Population limit percentage", default=0.9, type=float)
parser.add_argument("--train_size", "-tr",
                    help="Train size in %", default=0.001, type=float)
parser.add_argument("--test_size", "-te",
                    help="Test size in %", default=0.0002, type=float)

args = parser.parse_args()