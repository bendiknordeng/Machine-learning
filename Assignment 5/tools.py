import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

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
            target.append(letter) # save targets alphanumerically
    return np.array(images),np.array(data),np.array(target)

def scale(data):
    scaler = StandardScaler()
    scaler.fit(data) 
    return scaler, scaler.transform(data)

def reduce(data, n):
    pca = PCA(n_components = n)
    pca.fit(data)
    return pca, pca.transform(data)

# top_predictions returns the top-n predictions given an image and a classifier
def top_predictions(clf, frd, image, n):
    preds = clf.predict_proba([frd[image]])
    preds_idx = np.argsort(-preds, axis = 1)
    probs = []
    for p in preds_idx[0][:n]:
        probs.append([clf.classes_[p], preds[0][p]])
    return probs

# reshapes image data so that it can be processed by classifier
def flatten_img(image):
    return [float(i) for line in image for i in line]

# checks if a window in the sliding window algorithm is necessary to process
def bad_image(img):
    # All white = bad
    if np.sum(img) == 255 * len(img):
        return True

    # Keeps track of white pixels in a given column or row
    column_whites = np.zeros(20)
    row_whites = np.zeros(20)

    col_seq_length = 0
    row_seq_length = 0

    # Check if there is any pair of columns that are mostly white
    for i in range(20):
        column_whites[i] = sum([img[j] == 255 for j in range(i, 400, 20)]) >= 15
        if column_whites[i]:
            col_seq_length += 1
            if col_seq_length > 0:
                return True
        else:
            col_seq_length = 0

    # Check if there is any pair of rows that are mostly white
    for i in range(20):
        row_whites[i] = sum([p == 255 for p in img[40*i:40*(i+1)]]) >= 15
        if row_whites[i]:
            row_seq_length += 1
            if row_seq_length > 0:
                return True
        else:
            row_seq_length = 0

    return False


def sliding_window(clf, image, scaler, pca, window_size=20, stride=1):
    predictions = dict()
    image_x, image_y = image.shape
    ignore_to_x = False
    reset_y = 41
    
    # Ignore outlying windows (known to be white given detection images)
    for y in range(20, image_y-2*window_size, stride):
        for x in range(20, image_x-window_size, stride):

            # Flags to ignore if we have set a box already
            if ignore_to_x and ignore_to_x > x:
                if reset_y == y:
                    ignore_to_x = False
                else:
                    continue
            elif ignore_to_x:
                ignore_to_x = False

            # Extracts a window by using cropping
            cropp = image[x:x+window_size,y:y+window_size]
            cropped_image = np.reshape(cropp, window_size*window_size)

            # Check this box to see if should be classified
            if bad_image(cropped_image):
                continue
 
            # Make prediction from model
            img = scaler.transform([cropped_image])
            img = pca.transform(img)[0]
            p = top_predictions(clf,[img],0,1)[0]
            char = p[0]

            # Check if above threshold for classification
            ignore_to_x = x + 20
            reset_y = y + 1
            predictions[(x, y)] = char
    
    return predictions

def plot_prediction_windows(image, predictions, plot_size=(5,5), window_size=20):
    fig,ax = plt.subplots(1, figsize=plot_size)
    ax.imshow(image, cmap='gray', interpolation="nearest")
    
    for xy, c in predictions.items():
        rect = patches.Rectangle((xy[1]-1,xy[0]-1),window_size,window_size,linewidth=1,edgecolor='g',facecolor='none')
        ax.add_patch(rect)
        ax.text(xy[1]+window_size/2-1, xy[0]+30, c)

    plt.show()
    
def plot_confusion_matrix(y_true, y_pred, classes, title, cmap=plt.cm.Spectral):
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    plt.show()

def SVM(X_train, y_train):
    # Define model with optimal hyperparameters found from gridsearch
    clf = svm.SVC(gamma = 0.001, C=10, probability=True)
    clf.fit(X_train, y_train)
    return clf

def ANN(X_train, y_train):
    # Define model with optimal hyperparameters found from gridsearch
    clf = MLPClassifier(hidden_layer_sizes=1000,alpha=0.01, max_iter=1000)
    clf.fit(X_train, y_train)
    return clf