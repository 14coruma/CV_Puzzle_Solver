#!/usr/bin/python3
import sys

import cv2 as cv

if __name__ == "__main__":
    if len(sys.argv) < 2: exit("ERROR: Please provide a class label")
    label = sys.argv[1]
    i = int(sys.argv[2]) if len(sys.argv) >= 3 else 0
    
    # Video Capture code taken from https://stackoverflow.com/questions/604749/how-do-i-access-my-webcam-in-python
    cv.namedWindow("preview")
    vc = cv.VideoCapture(0)
    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False

    while rval:
        # Resize frame
        center = frame.shape[0] / 2, frame.shape[1] / 2
        size = min(frame.shape[0], frame.shape[1])
        x,y = center[1] - size/2, center[0] - size/2
        frame = frame[int(y):int(y+size), int(x):int(x+size)]
        frame = cv.resize(frame, (256,256))

        # Show frame and save image
        cv.imshow("preview", frame)
        cv.imwrite("./data/{0}/{0}_{1}.png".format(label, i), frame)
        i += 1

        # Get next frame (check for ESC key)
        rval, frame = vc.read()
        key = cv.waitKey(200)
        if key == 27: break # exit on ESC
    cv.destroyWindow("preview")
