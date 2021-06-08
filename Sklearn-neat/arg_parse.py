import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--population_size", "-p",
                    help="Number of individuals in the population", default=10, type=int)
parser.add_argument("--generations", "-g",
                    help="Number of generations", default=10, type=int)
parser.add_argument("--fitness_limit", "-l",
                    help="Fitness limit percentage", default=0.9, type=float)
parser.add_argument("--train_size", "-tr",
                    help="Train size in %", default=0.0002, type=float)
parser.add_argument("--test_size", "-te",
                    help="Test size in %", default=0.0002, type=float)

args, unknown = parser.parse_known_args()