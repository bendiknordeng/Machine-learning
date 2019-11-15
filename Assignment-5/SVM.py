#!/usr/bin/env python
# coding: utf-8

import tools
images, data, target = tools.load_data()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=10)

scaler, scaled_X_train = tools.scale(X_train)
pca, reduced_X_train = tools.reduce(scaled_X_train, 40)

from sklearn import svm

from sklearn.model_selection import GridSearchCV
params = {'C': [pow(10,x) for x in range(-1,3)], 'gamma': [0.01,0.001, 0.0001, 0.00001]}

cv = GridSearchCV(param_grid=params, estimator=svm.SVC(random_state=10), cv=10, n_jobs=-1)
cv.fit(reduced_X_train, y_train)

print("Train accuracy:", cv.score(reduced_X_train, y_train))

scaled_X_test = scaler.transform(X_test)
reduced_X_test = pca.transform(scaled_X_test)

print("Test accuracy:", cv.score(reduced_X_test, y_test))

print(cv.best_params_)

clf = svm.SVC(gamma = 0.001, C=10, probability=True)
clf.fit(reduced_X_train, y_train)
print("Train accuracy", clf.score(reduced_X_train, y_train))
print("Test accuracy:",clf.score(reduced_X_test, y_test))

bad_images = []
for i in range(len(X_test)):
    if clf.predict([reduced_X_test[i]])[0] != y_test[i]:
        bad_images.append(i)
print("Wrongly classified images:",bad_images)

import matplotlib.pyplot as plt
import numpy as np
X_test = [np.reshape(X_test[i], [20,20]) for i in range(len(X_test))]
fig, axs = plt.subplots(2,3, figsize = (8,6))
example_images = [0, 101, 300, 619, 1000, 1212]
img = example_images[0]

pred = clf.predict([reduced_X_test[img]])
axs[0,0].set_title('Prediction: ' + pred[0].capitalize())
axs[0,0].imshow(X_test[img])

img = example_images[1]
pred = clf.predict([reduced_X_test[img]])
axs[0,1].set_title('Prediction: ' + pred[0].capitalize())
axs[0,1].imshow(X_test[img])

img = example_images[2]
pred = clf.predict([reduced_X_test[img]])
axs[0,2].set_title('Prediction: ' + pred[0].capitalize())
axs[0,2].imshow(X_test[img])

img = example_images[3]
pred = clf.predict([reduced_X_test[img]])
axs[1,0].set_title('Prediction: ' + pred[0].capitalize())
axs[1,0].imshow(X_test[img])

img = example_images[4]
pred = clf.predict([reduced_X_test[img]])
axs[1,1].set_title('Prediction: ' + pred[0].capitalize())
axs[1,1].imshow(X_test[img])

img = example_images[5]
pred = clf.predict([reduced_X_test[img]])
axs[1,2].set_title('Prediction: ' + pred[0].capitalize())
axs[1,2].imshow(X_test[img])

plt.show()

tools.plot_confusion_matrix(y_test, clf.predict(reduced_X_test), clf.classes_, "Confusion matrix")
