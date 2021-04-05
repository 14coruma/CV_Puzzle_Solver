#!/usr/bin/python3
import math
import numpy as np
import os

from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import cv2 as cv

def image_name_and_class(loc):
    images, labels = [], []
    for label in os.listdir(loc):
        path = os.path.join(loc, label)
        images += [cv.imread(os.path.join(path, x), 0) for x in os.listdir(path)]
        labels += [label] * len(os.listdir(path))
    return images, labels

def sift_features(images):
    descripts = []
    sift = cv.SIFT_create()
    for img in images:
        keypts, des = sift.detectAndCompute(img, None)
        descripts.append(des)
    return descripts

def cluster_words(k, descripts):
    #  kmeans = KMeans(n_clusters=k, random_state=0).fit(descripts)
    #  return kmeans.cluster_centers_
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv.KMEANS_RANDOM_CENTERS
    compactness, _, centers = cv.kmeans(
            np.array(descripts), k, None, criteria, 10, flags)
    radius = math.sqrt(compactness / len(descripts))
    return centers, radius

def build_bovw(images, words, thresh):
    descripts = sift_features(images)
    bags = []
    for img, des in zip(images, descripts):
        hist = [0] * len(words)
        for feature in des:
            for i in range(len(words)):
                if distance.euclidean(feature, words[i]) < thresh:
                    hist[i] += 1
        bags.append(hist)
    return bags

def NN(X_test, X_train, y_train):
    y_pred = []
    for test_bovw in X_test:
        best_class = None
        best_dist = float('inf')
        for train_bovw, y in zip(X_train, y_train):
            dist = distance.euclidean(train_bovw, test_bovw)
            if dist < best_dist:
                best_dist = dist
                best_class = y
        y_pred.append(best_class)
    return y_pred

if __name__ == "__main__":
    # Grab data
    img, y = image_name_and_class('./data')
    enc = LabelEncoder()
    y = enc.fit_transform(y)
    img_train, img_test, y_train, y_test = train_test_split(
            img, y, test_size=0.33, random_state=0, stratify=y)

    #  # Train classifier
    descripts = sift_features(img_train)
    visual_words, radius = cluster_words(100, [val for row in descripts for val in row])
    X_train = build_bovw(img_train, visual_words, radius)
  
    # Test classifier
    X_test = build_bovw(img_test, visual_words, radius) 
    y_pred = NN(X_test, X_train, y_train)

    # Evaluate model
    print("Accuracy: {}".format(accuracy_score(y_test, y_pred)))
    print("Confusion Matrix:\n{}".format(confusion_matrix(y_test, y_pred)))

    img = [cv.imread('test2.png', 0)]
    print(enc.inverse_transform(NN(build_bovw(img, visual_words, radius), np.concatenate((X_train, X_test), axis=0), np.concatenate((y_train, y_test), axis=0))))
