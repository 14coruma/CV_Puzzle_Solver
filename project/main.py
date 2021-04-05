#!/usr/bin/python3
import cv2 as cv

if __name__ == "__main__":
    # BEGIN: from https://stackoverflow.com/questions/604749/how-do-i-access-my-webcam-in-python
    cv.namedWindow("preview")
    vc = cv.VideoCapture(0)
    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False

    while rval:
        cv.imshow("preview", frame)
        rval, frame = vc.read()
        key = cv.waitKey(20)
        if key == 27: break # exit on ESC
    cv.destroyWindow("preview")
    # END from https://stackoverflow.com/questions/604749/how-do-i-access-my-webcam-in-python
