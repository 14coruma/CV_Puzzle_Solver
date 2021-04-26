#!/usr/bin/python3
import argparse
import numpy as np
import cv2 as cv

from tensorflow.keras import models

import Classifier
from AI import Sudoku_AI
from Features import Sudoku_Features

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str,
        help="filename of the image of a puzzle you would like to solve")
    return parser.parse_args()

def get_puzzle_type(image):
    '''
    '''
    # Resize image
    h, w = image.shape[0], image.shape[1]
    center = h/2, w/2
    size = min(h, w)
    x,y = center[1] - size/2, center[0] - size/2
    image_resized = image[int(y):int(y+size), int(x):int(x+size)]
    image_resized = cv.resize(image_resized, (256,256))
    # Predict puzzle type
    ptype = Classifier.predict(class_model, image_resized)

    return ptype

def get_features(ptype, ocr_model, image):
    features = {'image': image}
    if ptype == 'sudoku':
        cropped = Sudoku_Features.locate_puzzle(cv.cvtColor(image, cv.COLOR_BGR2GRAY))
        board = Sudoku_Features.construct_board(cropped, ocr_model)
        visualized = Sudoku_Features.visualize_board(cropped, board)
        cv.imshow('Sudoku features', visualized)
        features['cropped'] = cropped
        features['board'] = board
    return features

def find_solution(ptype, features):
    solution = None
    if ptype == 'sudoku':
        solution = Sudoku_AI.solve(features['board'])
        visualized = Sudoku_Features.visualize_board(features['cropped'], solution)
        cv.imshow('Sudoku solved', visualized)
    return solution

if __name__ == "__main__":
    args = get_args()
    # Load ML models
    # (Do one dummy classification for TF models, to make sure they load fully)
    class_model = Classifier.train('./Data')
    ocr_model = models.load_model('Models/OCR_CNN_Trained')
    ocr_model.predict(np.array(np.zeros((1,28,28,1))))

    image = cv.imread(args.filename, cv.IMREAD_UNCHANGED)
    cv.imshow('Image', image)

    ptype = get_puzzle_type(image)
    print("Detected puzzle type: ", ptype)

    features = get_features(ptype, ocr_model, image)
    print("Detected puzzle features")

    solution = find_solution(ptype, features)
    print("Found solution:\n", solution)

    cv.waitKey()