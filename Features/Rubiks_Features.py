import math
import numpy as np
import sys

import cv2 as cv

if __name__ == "__main__":
    rubiks_image_folder = "Images/rubiks_0/"
    debug = True
    #if len(sys.argv) > 1: filename = sys.argv[1]
    filename = "Images/rubiks_0/rubiks_0_2.jpg"
    img = cv.imread(filename)

    # Threshold image for very colorful or white pixels (like you'd expect on a rubik's cube)
    blurred = cv.GaussianBlur(img, (3,3), cv.BORDER_REFLECT)
    hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)
    color = cv.inRange(hsv, (0,150,150), (180,255,255))
    white = cv.inRange(hsv, (0,0,200), (180,10,255))
    thresh = cv.bitwise_or(color, white)
    cv.imshow("color", thresh)
    cv.waitKey()

    # Edge detection
    blurred = cv.GaussianBlur(thresh, (3,3), cv.BORDER_REFLECT)
    edges = cv.Canny(blurred, 25, 250)
    cv.imshow("edges", edges)

    # Get contours
    contours,_ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Find 9 biggest, most squarelike, contours in image
    squares = []
    for contour in contours:
        # Ref: https://stackoverflow.com/questions/18207181/opencv-python-draw-minarearect-rotatedrect-not-implemented
        # Ref: https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html
        # Find bounding rect
        rect = cv.minAreaRect(contour)
        box = cv.boxPoints(rect)
        box = np.int0(box)
        cnt_area, box_area = cv.contourArea(contour), cv.contourArea(box)
        # If contour area is not much smaller than box area, then it is pretty square
        if cnt_area > .9 * box_area:
            squares.append(box)
    squares = sorted(squares, key=cv.contourArea)[-9:]

    cv.drawContours(img, squares, -1, (0,0,255), 2)
    cv.imshow("preview", img)
    cv.waitKey()