import pandas as pd
import numpy as np
import random
import math
from sklearn.model_selection import train_test_split

def distance(instance1, instance2, length):
    d = 0
    for i in range(length):
        d += (instance1[i]-instance2[i])**2
    return math.sqrt(d)

def get_neighbours(instance, X_train, y_train, k):
    distances = []
    for i in range(len(X_train)):
        d = distance(instance,X_train[i],len(X_train[i]))
        distances.append(d)
    idx = np.argsort(distances)[1:k+1]
    neighbours = []
    classes = []
    for index in idx:
        classes.append(y_train[index])
    return idx, classes

def get_votes(neighbours, classes):
    votes = list(np.zeros(classes))
    for i in neighbours:
        votes[i] += 1
    return votes.index(max(votes))

def regression(neighbours):
    return np.mean(neighbours)

def predict(X_test, X_train, y_train, k, alg):
    predictions = []
    for i in X_test:
        idx, classes = get_neighbours(i, X_train, y_train, k)
        if alg=='c':
            prediction = get_votes(classes, len(set(y_train)))
        if alg=='r':
            prediction = regression(classes)
        predictions.append(prediction)
    return predictions, idx

def get_accuracy(predictions, y_test, alg):
    if alg=='c':
        correct = 0
        for i in range(len(predictions)):
            if predictions[i] == y_test[i]:
                correct += 1
        print('Accuracy:',float(correct/len(predictions)))
    if alg=='r':
        absolute_error = 0
        square_error = 0
        for i in range(len(predictions)):
            absolute_error += abs(predictions[i]-y_test[i])
            square_error += (predictions[i]-y_test[i])**2
        print('MAE:', absolute_error/len(predictions))
        print('MSE:', square_error/len(predictions))

def knn(alg):
    if alg=='c':
        df = pd.read_csv('data/knn_classification.csv')
    if alg=='r':
        df = pd.read_csv('data/knn_regression.csv')
    k = 10
    y = df['y']
    X = df.drop(['y'], axis=1)
    rs = 10

    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.33, random_state=rs)

    predictions, idx = predict(X_test, X_train, y_train, k, alg)
    get_accuracy(predictions, y_test, alg)

    task_instance = [list(X.iloc[123])]
    prediction, idx = predict(task_instance, X_train, y_train, k, alg)
    print("Instance 124:", task_instance[0])
    print("Actual class:", y.iloc[123])
    print("Predicted class:", prediction)
    print("Neighbours:")
    for index in idx:
        print(X_train[index], "Class:", y_train[index])

print('k-NN classification')
knn('c') #knn_classification
print('--------------------------')
print('k-NN regression')
knn('r') #knn_regression
