import sys
import os

from matplotlib import pyplot as plt
# from sklearn.datasets import make_classification
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import cifar10
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.utils import to_categorical
import numpy as np
import cv2
import math
import sys
import logging
from io import StringIO 
import re
from arg_parse import *
from sklearn.model_selection import StratifiedShuffleSplit
import seaborn as sns
import contextlib
import datetime

logging.getLogger('matplotlib.font_manager').disabled = True

import seaborn as sns

sns.set_style('whitegrid')

sys.path.append(os.getcwd()+"/Sklearn-neat")

import neat
from neat import math_util
from neat.puissance import Puissance 

from neuro_evolution import NEATClassifier

"""
Custom logging
"""
root = logging.getLogger()
root.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)
"""
End
"""

now = datetime.datetime.now() # current date and time
time = now.strftime("%d.%m_%H.%M")

output_folder = "outputs/output_"+time+"/"

fig_loc = "figures/"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    
os.chdir(output_folder)
    
print("Current working directory: {}".format(os.getcwd()))

logfile = open('output.txt', 'w')

original_stderr = sys.stderr
original_stdout = sys.stdout

class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() # If you want the output to be visible immediately
    def flush(self) :
        for f in self.files:
            f.flush()

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

X = np.append(x_train,x_test).reshape(60000,32,32,3)
y = np.append(y_train,y_test).reshape(60000,1)

assert X.shape == (60000, 32, 32, 3)
assert y.shape == (60000, 1)

#Preprocess the data
X = X.astype('float32')
X /= 255

sss = StratifiedShuffleSplit(n_splits=5, 
                             train_size=args.train_size, 
                             test_size=args.test_size,
                             random_state=0)

for train_index, test_index in sss.split(X, y):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
print(X_train.shape)
print(y_train.shape)

print("---")

print(X_test.shape)
print(y_test.shape)
    
def find_metric_in_output(output, string):    
    result = re.findall(r"\b"+string+r" ([0-9]+\.[0-9]+)\b", str(output))
    
    result = np.array(result).astype('float')
        
    return result 

def find_puissance_in_output(output):    
    result = re.findall(r"\b(?<=Unique puissance values: {).*?(?=})\b", str(output))    
    return result 
    
X_train_fl = X_train.reshape((X_train.shape[0], -1))
X_test_fl = X_test.reshape((X_test.shape[0], -1))

try:
    from neat.puissance import Puissance

    puissance_config = Puissance()

    clf = NEATClassifier(number_of_generations=args.generations,
                         fitness_threshold=args.fitness_limit,
                         pop_size=args.population_size,
                         puissance_config = puissance_config)
    
    logging.info("Running NEAT puissance")
    
except:
    clf = NEATClassifier(number_of_generations=args.generations,
                         fitness_threshold=args.fitness_limit,
                         pop_size=args.population_size)
    
    logging.info("Running NEAT")
    
"""
Training the NEAT model
"""
sys.stdout = Tee(sys.stdout, logfile)
sys.stderr = sys.stdout
    
neat_genome = clf.fit(X_train_fl, y_train.ravel())

sys.stdout = original_stdout
sys.stderr = original_stderr
logfile.close()
    
y_pred = neat_genome.predict(X_test_fl)
    
print(classification_report(y_test.ravel(), y_pred.ravel()))

"""
Reading the output and processing results
"""
output = open(output_loc+'output.txt', "r").read()

gen_time = find_metric_in_output(output, "Generation time:")
cum_gen_time = np.array([])

for i in range(1,len(gen_time)+1):
    cum_gen_time = np.append(cum_gen_time, gen_time[:i].sum())
    
results = {}

results['best_fitness'] = find_metric_in_output(output, "Best fitness:")
results['avg_adj_fitness'] = find_metric_in_output(output, "Average adjusted fitness:")
results['pop_avg_fitness'] = find_metric_in_output(output, "Population's average fitness:")
results['gen_time'] = gen_time
results['cum_gen_time'] = cum_gen_time
results['stdev'] = find_metric_in_output(output, "stdev:")
# metrics['puissance'] = puissance = find_puissance_in_output(output)

assert len(results['best_fitness']) == \
        len(results['avg_adj_fitness']) == \
        len(results['pop_avg_fitness']) == \
        len(results['gen_time']) == \
        len(results['stdev'])

"""
Plotting the results
"""
def plot_results(plots, xlabel, ylabel, fig_name):
    fig,ax = plt.subplots(figsize=(15,8))

    for i in range(0,len(plots)):
        plt.plot(plots[i]['x'],
                 plots[i]['y'],
                 label = plots[i]['label'])

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.legend()

    plt.xticks(np.arange(0, args.generations, math.ceil(args.generations/100)))

    plt.savefig(fig_loc+fig_name)

    plt.plot()
    
    
"""
Fitness
"""
plots = {}

plots[0] = {
    "x":np.linspace(0,args.generations,args.generations),
    "y":results['avg_adj_fitness'],
    "label":"Average adjusted fitness"
}

plots[1] = {
    "x":np.linspace(0,args.generations,args.generations),
    "y":results['pop_avg_fitness'],
    "label":"Population's average fitness"  
}

plots[2] = {
    "x":np.linspace(0,args.generations,args.generations),
    "y":results['best_fitness'],
    "label":"Best fitness"  
}

plot_results(plots, "Generation", "Fitness", "fitness.png")

"""
Standard deviation
"""
plots = {}

plots[0] = {
    "x":np.linspace(0,args.generations,args.generations),
    "y":results['stdev'],
    "label":"Standard deviation"
}

plot_results(plots, "Generation", "Standard deviation", "stdev.png")

"""
Generaion time
"""
plots = {}

plots[0] = {
    "x":np.linspace(0,args.generations,args.generations),
    "y":results['gen_time'],
    "label":"Generation time (seconds)"
}

plot_results(plots, "Generation", "Generation time (seconds)", "gen_time.png")

"""
Cumulative generaion time
"""
plots = {}

plots[0] = {
    "x":np.linspace(0,args.generations,args.generations),
    "y":results['cum_gen_time'],
    "label":"Cumulative generation time (seconds)"
}

plot_results(plots, "Generation", "Cumulative generation time (seconds)", "cum_gen_time.png")
