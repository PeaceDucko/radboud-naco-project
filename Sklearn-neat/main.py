"""
============================
Plotting NEAT Classifier
============================

An example plot of :class:`neuro_evolution._neat.NEATClassifier`
"""
from matplotlib import pyplot as plt
# from sklearn.datasets import make_classification
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from neuro_evolution import NEATClassifier
from tensorflow.keras.datasets import cifar10
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.utils import to_categorical
import numpy as np
import cv2
import sys
import logging
from io import StringIO 
import re
from arg_parse import *
from sklearn.model_selection import StratifiedShuffleSplit

root = logging.getLogger()
root.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)

class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout
        
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

X = np.append(x_train,x_test).reshape(60000,32,32,3)
y = np.append(y_train,y_test).reshape(60000,1)

assert X.shape == (60000, 32, 32, 3)
assert y.shape == (60000, 1)

#Preprocess the data
X = X.astype('float32')
X /= 255

sss = StratifiedShuffleSplit(n_splits=5, 
                             train_size=0.015, # 900 train 
                             test_size=0.003, # 180 test
                             random_state=0)

for train_index, test_index in sss.split(X, y):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
with Capturing() as output:
    X_train_fl = X_train.reshape((X_train.shape[0], -1))
    X_test_fl = X_test.reshape((X_test.shape[0], -1))


    clf = NEATClassifier(number_of_generations=args.generations,
                         fitness_threshold=args.population_limit,
                         pop_size=args.population_size)

    neat_genome = clf.fit(X_train_fl, y_train.ravel())
    y_pred = neat_genome.predict(X_test_fl)
    
    print(np.array(re.findall(r'\bGeneration ([0-9]+\.[0-9]+)\b', str(output))))
    
print(classification_report(y_test.ravel(), y_pred.ravel()))

best_fitness = np.array(re.findall(r'\bBest fitness: ([0-9]+\.[0-9]+)\b', str(output))).astype('float')
avg_adj_fitness = np.array(re.findall(r'\bAverage adjusted fitness: ([0-9]+\.[0-9]+)\b', str(output))).astype('float')

fig,ax = plt.subplots(figsize=(15,8))

plt.plot(np.linspace(1,len(best_fitness),len(best_fitness)),
         best_fitness,
         label="Best fitness")
plt.plot(np.linspace(1,len(avg_adj_fitness),len(avg_adj_fitness)),
         avg_adj_fitness,
         label="Average adjusted fitness")

plt.xlabel("Generation")
plt.ylabel("Fitness")

plt.legend()

plt.xticks(np.linspace(1,len(best_fitness),len(best_fitness)))

plt.savefig("figures/fitness.png")

plt.plot()

gen_time = np.array(re.findall(r'\bGeneration time: ([0-9]+\.[0-9]+)\b', str(output))).astype('float')

fig,ax = plt.subplots(figsize=(15,8))

plt.plot(np.linspace(1,len(gen_time),len(gen_time)),gen_time)

plt.xlabel("Generation")
plt.ylabel("Generation time (seconds)")

plt.xticks(np.linspace(1,len(gen_time),len(gen_time)))

plt.savefig("figures/generation_time.png")

plt.plot()
