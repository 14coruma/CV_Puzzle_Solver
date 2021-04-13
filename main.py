#!/usr/bin/python3
import numpy as np
import cv2 as cv

import Classifier
from AI import Sudoku

if __name__ == "__main__":
    model = Classifier.train('./data')

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
        frame = frame[int(y):int(y+size), int(x):int(x+size)]
        frame = cv.resize(frame, (256,256))
        # Predict puzzle type
        label = Classifier.predict(model, frame)

        # Must see new puzzle type a couple times in a row to be sure
        # TODO: Maybe recurrent or markov strategy better here?
        if label == new_label: new_label_count += 1
        else: new_label_count, new_label = 0, label
        if new_label_count > 3: curr_label = label

        # Display matched features
        img2 = None
        if curr_label != "None":
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            img = cv.imread('./data/{0}/{0}_0.png'.format(curr_label), 0)
            orb = cv.ORB_create()
            kp1, des1 = orb.detectAndCompute(frame, None)
            kp2, des2 = orb.detectAndCompute(img, None)
            bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key = lambda x: x.distance)
            img2 = cv.drawMatches(frame, kp1, img, kp2, matches[:10], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        else: img2 = frame

        cv.imshow("preview", img2)
        rval, frame = vc.read()
        key = cv.waitKey(200)
        if key == 27: break # exit on ESC
    cv.destroyWindow("preview")
