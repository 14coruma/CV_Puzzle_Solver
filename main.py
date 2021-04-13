#!/usr/bin/python3
import numpy as np
import cv2 as cv

from tensorflow.keras import models

import Classifier
from AI import Sudoku_AI
from Features import Sudoku_Features

if __name__ == "__main__":
    # Load ML models
    # (Do one dummy classification for TF models, to make sure they load fully)
    class_model = Classifier.train('./Data')
    ocr_model = models.load_model('Models/OCR_CNN_Trained')
    ocr_model.predict(np.array(np.zeros((1,28,28,1))))

    # Video Capture code taken from https://stackoverflow.com/questions/604749/how-do-i-access-my-webcam-in-python
    cv.namedWindow("preview")
    vc = cv.VideoCapture(0)
    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False

    curr_label = 'rubiks'
    new_label = 'rubiks'
    new_label_count = 0
    while rval:
        # Resize frame
        center = frame.shape[0] / 2, frame.shape[1] / 2
        size = min(frame.shape[0], frame.shape[1])
        x,y = center[1] - size/2, center[0] - size/2
        frame2 = frame[int(y):int(y+size), int(x):int(x+size)]
        frame2 = cv.resize(frame2, (256,256))
        # Predict puzzle type
        label = Classifier.predict(class_model, frame2)

        # Must see new puzzle type a couple times in a row to be sure
        # TODO: Maybe recurrent or markov strategy better here?
        if label == new_label: new_label_count += 1
        else: new_label_count, new_label = 0, label
        if new_label_count > 3: curr_label = label

        # Display matched features
        img_out = None
        if curr_label == "sudoku":
            cropped = Sudoku_Features.locate_puzzle(cv.cvtColor(frame, cv.COLOR_BGR2GRAY))
            board = Sudoku_Features.construct_board(cropped, ocr_model) 
            visualized = Sudoku_Features.visualize_board(cropped, board)
            img_out = visualized

        elif curr_label != "None":
            frame2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
            img = cv.imread('./Data/{0}/{0}_0.png'.format(curr_label), 0)
            orb = cv.ORB_create()
            kp1, des1 = orb.detectAndCompute(frame2, None)
            kp2, des2 = orb.detectAndCompute(img, None)
            bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key = lambda x: x.distance)
            img_out = cv.drawMatches(frame2, kp1, img, kp2, matches[:10], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        else: img_out = frame2

        cv.imshow("preview", img_out)

        rval, frame = vc.read()
        key = cv.waitKey(200)
        if key == 27: break # exit on ESC

    cv.destroyWindow("preview")
