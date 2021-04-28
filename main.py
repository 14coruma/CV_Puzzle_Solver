#!/usr/bin/python3
import argparse
import numpy as np
import cv2 as cv

from tensorflow.keras import models

import Classifier
from AI import Akari_AI, Sudoku_AI, Rubiks_AI
from Features import Akari_Features, Sudoku_Features, Rubiks_Features

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
        board = Sudoku_Features.construct_board(cropped, ocr_model, allow_zeros=False)
        visualized = Sudoku_Features.visualize_board(cropped, board)
        cv.imshow('Sudoku features', visualized)
        features['cropped'] = cropped
        features['board'] = board
    elif ptype == 'akari':
        cropped = Akari_Features.locate_puzzle(cv.cvtColor(image, cv.COLOR_BGR2GRAY))
        board = Akari_Features.construct_board(cropped, ocr_model, allow_zeros=True)
        visualized = Akari_Features.visualize_board(cropped, board)
        cv.imshow('Akari features', visualized)
        features['cropped'] = cropped
        features['board'] = board
    elif ptype == 'rubiks':
        faces = []
        print("Need to see each side of cube...")
        for side in ['Front', 'Right', 'Back', 'Left', 'Upper', 'Down']:
            filename = input("  {} filename: ".format(side))
            faces.append(Rubiks_Features.face_from_filename(filename))
        features['faces'] = faces
    return features

def find_solution(ptype, features):
    solution = None
    if ptype == 'sudoku':
        solution = Sudoku_AI.solve(features['board'])
        visualized = Sudoku_Features.visualize_board(features['cropped'], solution)
        cv.imshow('Sudoku solved', visualized)
        cv.waitKey()
    elif ptype == 'akari':
        solution = Akari_AI.solve(features['board'])
        visualized = Akari_Features.visualize_board(features['cropped'], solution)
        cv.imshow('Akari solved', visualized)
        cv.waitKey()
    elif ptype == 'rubiks':
        faceletstr = Rubiks_AI.faceletstr_from_faces(features['faces'])
        solution = Rubiks_AI.solve(faceletstr)
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
    cv.waitKey()

    print("Detecting puzzle type...")
    ptype = get_puzzle_type(image)
    print("Found puzzle type: ", ptype)

    print("Detecting puzzle features...")
    features = get_features(ptype, ocr_model, image)
    print("Found puzzle features.")

    print("Solving puzzle...")
    solution = find_solution(ptype, features)
    print("Found solution:\n", solution)