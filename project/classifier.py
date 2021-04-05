#!/usr/bin/python3
import math
from numba import jit, njit
import numpy as np
import os

from scipy.spatial import distance

from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

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
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv.KMEANS_RANDOM_CENTERS
    compactness, _, centers = cv.kmeans(
            np.array(descripts), k, None, criteria, 10, flags)
    radius = math.sqrt(compactness / len(descripts))
    return centers, radius

def closest_word(feature, words):
    min_dist, best_i = float('inf'), 0
    for i in range(len(words)):
        dist = distance.euclidean(feature, words[i])
        if (dist < min_dist):
            best_i, min_dist = i, dist
    return best_i

def build_bovw(images, words, thresh):
    descripts = sift_features(images)
    bags = []
    for img, des in zip(images, descripts):
        hist = [0] * len(words)
        for feature in des:
            i = closest_word(feature, words)
            hist[i] += 1
        bags.append(hist)
    return bags

def KNN(X_test, X_train, y_train, k=1):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train) 
    return knn.predict(X_test), knn.predict_proba(X_test)

def NB(X_test, X_train, y_train):
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)
    prob = gnb.predict_log_proba(X_test)
    return y_pred, [max(prob[i]) for i in range(len(y_pred))]

def train(data_loc):
    # Grab data
    img, y = image_name_and_class('./data')
    enc = LabelEncoder()
    y = enc.fit_transform(['None'] + y)[1:]
    # Train classifier
    descripts = sift_features(img)
    visual_words, radius = cluster_words(100, [val for row in descripts for val in row])
    X = build_bovw(img, visual_words, radius)
    return {"visual_words": visual_words, "radius": radius, "data": X, "labels": y, "enc": enc}

def predict(model, img):
    X = build_bovw([img], model["visual_words"], model["radius"])
    pred, prob = KNN(X, model["data"], model["labels"])
    return model["enc"].inverse_transform(pred)[0]


if __name__ == "__main__":
    # Grab data
    print("Grabbing images")
    img, y = image_name_and_class('./data')
    enc = LabelEncoder()
    y = enc.fit_transform(['None'] + y)[1:]
    img_train, img_test, y_train, y_test = train_test_split(
            img, y, test_size=0.33, random_state=0, stratify=y)

    # Train classifier
    print("SIFT")
    descripts = sift_features(img_train)
    print("Cluster")
    visual_words, radius = cluster_words(100, [val for row in descripts for val in row])
    print("Hist")
    X_train = build_bovw(img_train, visual_words, radius)
  
    # Test classifier
    print("Hist2")
    X_test = build_bovw(img_test, visual_words, radius) 
    print("Bayes")
    y_pred, y_prob = KNN(X_test, X_train, y_train)
    print(y_prob)

    # Evaluate model
    print("Accuracy: {}".format(accuracy_score(y_test, y_pred)))
    print("Confusion Matrix:\n{}".format(confusion_matrix(y_test, y_pred)))

    img = [cv.imread('test2.png', 0)]
    pred, prob = KNN(build_bovw(img, visual_words, radius), np.concatenate((X_train, X_test), axis=0), np.concatenate((y_train, y_test), axis=0))
    print(enc.inverse_transform(pred), prob)
