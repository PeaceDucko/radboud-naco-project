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
from keras.datasets import cifar10
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.utils import to_categorical
import numpy as np
import cv2

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
assert x_train.shape == (50000, 32, 32, 3)
assert x_test.shape == (10000, 32, 32, 3)
assert y_train.shape == (50000, 1)
assert y_test.shape == (10000, 1)

#Transform (50000, 32, 32, 3) to (50000, 32, 32) with grayscaling.
# x_train = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in x_train])
# x_test = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in x_test])

#Preprocess the data
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
# y_train = to_categorical(y_train, 10)
# y_test = to_categorical(y_test, 10)

# one_hot_encoder = OneHotEncoder(sparse=False)

# one_hot_encoder.fit(y_train)

# y_train = one_hot_encoder.transform(y_train)
# y_test = one_hot_encoder.transform(y_test)

# def testSet(x,y,val):
#     X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=val, random_state=42)
#     return X_test,Y_test

# x_test,y_test = testSet(x_test,y_test, 0.001)
# x_train,y_train = testSet(x_train,y_train, 0.001)

print(x_test.shape)
print(y_test.shape)

x_train_fl = x_train.reshape((x_train.shape[0], -1))
x_test_fl = x_test.reshape((x_test.shape[0], -1))


clf = NEATClassifier(number_of_generations=1,
                     fitness_threshold=0.90,
                     pop_size=150)

neat_genome = clf.fit(x_train_fl, y_train.ravel())
y_predicted = neat_genome.predict(x_test_fl)

fig = plt.figure()
ax = plt.axes(projection='3d')

#Data for three-dimensional scattered points
train_z_data = x_train_fl
train_x_data = x_train_fl[:, 1]
train_y_data = x_train_fl[:, 0]
ax.scatter3D(train_x_data, train_y_data, train_z_data, c='Blue')

test_z_data = y_predicted
test_x_data = x_test_fl[:, 1]
test_y_data = x_test_fl[:, 0]
ax.scatter3D(test_x_data, test_y_data, test_z_data, c='Red')
ax.legend(['Actual', 'Predicted'])
plt.show()

print(classification_report(y_test.ravel(), y_predicted.ravel()))