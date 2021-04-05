#!/usr/bin/python3
import numpy as np
import os

from scipy.spatial import distance
from sklearn.model_selection import train_test_split

import cv2 as cv

def image_name_and_class(loc):
    images, labels = [], []
    for label in os.listdir(loc):
        path = os.path.join(loc, label)
        images += [os.path.join(path, x) for x in os.listdir(path)]
        labels += [label] * len(os.listdir(path))
    return images, labels

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
    img_train, img_test, y_train, y_test = train_test_split(
            img, y, test_size=0.33, random_state=0, stratify=y)

    #  # Train classifier
    #  descrips = orb_features(img_train)
    #  visual_words = determine_words(k, descrips)
    #  X_train = build_bovw(img_train, visual_words)
#  
    #  # Test classifier
    #  X_test = build_bovw(img_test, visual_words) 
    #  y_pred = NN(X_test, X_train, y_train)
