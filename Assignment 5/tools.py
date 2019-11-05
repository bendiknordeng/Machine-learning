import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.neural_network import MLPClassifier

def load_data(): 
    images = []
    data = []
    target = []
    for i in range(97,123):
        letter = chr(i)
        dir = 'data/chars74k/'+letter
        for img in os.listdir(dir):
            image = plt.imread(dir+'/'+img)
            images.append(image)
            data.append([float(i) for line in image for i in line])
            target.append(letter)
    return np.array(images),np.array(data),np.array(target)

def scale(data):
    scaler = StandardScaler()
    scaler.fit(data)
    return scaler, scaler.transform(data)

def reduce(data, n):
    pca = PCA(n_components = n)
    pca.fit(data)
    return pca, pca.transform(data)

def top_predictions(clf, frd, image, n):
    preds = clf.predict_proba([frd[image]])
    preds_idx = np.argsort(-preds, axis = 1)
    probs = []
    for p in preds_idx[0][:n]:
        probs.append([clf.classes_[p], str(round(preds[0][p]*100,2))+'%'])
    return probs

def flatten_img(image):
    return [float(i) for line in image for i in line]

def sliding_window(image, stepSize, windowSize):
    windows = []
    flattened = []
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize[1]):
        for x in range(0, image.shape[1], stepSize[0]):
            window = img[y:(y+windowSize[1]),x:(x+windowSize[0])]
            flat = flatten_img(window)
            unique, counts = np.unique(flat, return_counts=True)
            if dict(zip(unique, counts))[255.0] < 250: # Large parts of window white
                windows.append(window)
                flattened.append(flat)
    return np.array(windows), np.array(flattened)

def SVM(X_train, y_train):
    clf = svm.SVC(gamma = 0.001, C=10, probability=True)
    clf.fit(X_train, y_train)
    return clf

def ANN(X_train, y_train):
    clf = MLPClassifier(hidden_layer_sizes=1000,alpha=0.01, max_iter=1000)
    clf.fit(X_train, y_train)
    return clf