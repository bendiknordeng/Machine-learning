#!/usr/bin/env python
# coding: utf-8

import numpy as np
import tools
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

images, data, target = tools.load_data()

scaler, data = tools.scale(data)
pca, feature_reduced_data = tools.reduce(data,40)

X_train, X_test, y_train, y_test = train_test_split(feature_reduced_data, target, test_size=0.2, random_state=10)
clf = tools.ANN(X_train,y_train)

img1 = plt.imread('data/detection-images/detection-1.jpg')
preds = tools.sliding_window(clf, img1, scaler, pca)
tools.plot_prediction_windows(img1, preds, (10,10))

img2 = plt.imread('data/detection-images/detection-2.jpg')
preds = tools.sliding_window(clf, img2, scaler, pca)
tools.plot_prediction_windows(img2, preds, (15,10))

