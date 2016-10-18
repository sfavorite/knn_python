#!/usr/bin/env python

# Create a classifier
import random
from scipy.spatial import distance


def euc(a, b):
    return distance.euclidean(a,b)

def cb(a, b):
    return distance.cityblock(a, b)

def mink(a, b):
    return distance.minkowski(a, b, 1)

class bareKNN():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []

        for row in X_test:
            #label = random.choice(self.y_train)
            label = self.closest(row)
            predictions.append(label)

        return predictions

    def closest(self, row):
        best_dist = mink(row, self.X_train[0])
        best_index = 0
        for i in range(3, len(self.X_train)):
            dist = cb(row, self.X_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i
        return self.y_train[best_index]


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .5)

my_classifier = bareKNN()
my_classifier.fit(X_train, y_train)

predictions = my_classifier.predict(X_test)
print(accuracy_score(y_test, predictions))
