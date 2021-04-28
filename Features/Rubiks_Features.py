import math
import numpy as np
import sys

import cv2 as cv
from scipy.spatial import KDTree


if __name__ == "__main__":
    rubiks_image_folder = "Images/rubiks_0/"
    #if len(sys.argv) > 1: filename = sys.argv[1]
    filename = "Images/rubiks_0/rubiks_0_1.jpg"
    img = cv.imread(filename)

    # Threshold image for very colorful or white pixels (like you'd expect on a rubik's cube)
    blurred = cv.GaussianBlur(img, (3,3), cv.BORDER_REFLECT)
    hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)
    color = cv.inRange(hsv, (0,100,150), (180,255,255)) # HSV Colors (any H, high S, high V)
    white = cv.inRange(hsv, (0,0,200), (180,10,255))    # HSV White (any H, low S, high V)
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

    #cv.drawContours(img, squares, -1, (0,0,255), 2)
    #cv.imshow("preview", img)
    #cv.waitKey()

    # Identify color of each square
    # Idea to use KDTree for nearest color lookup: https://medium.com/codex/rgb-to-color-names-in-python-the-robust-way-ec4a9d97a01f
    # List of possible rubik's cube BGR colors:
    bgr_names = ['red', 'green', 'blue', 'yellow', 'orange', 'white']
    bgr_colors = [
        [20,20,235],  # Red
        [20,235,20],  # Green
        [235,20,20],  # Blue
        [20,235,235], # Yellow
        [20,165,235], # Orange
        [235,235,235] # White
    ]
    kdt = KDTree(bgr_colors)

    colors = []
    for square in squares:
        # Ref: https://stackoverflow.com/questions/33234363/access-pixel-values-within-a-contour-boundary-using-opencv-in-python
        # Create a mask image that contains the contour filled in
        cimg = np.zeros_like(img)
        cv.drawContours(cimg, [square], 0, color=255, thickness=-1)
        pts = np.where(cimg == 255)
        # Find average color in img within this mask
        pixels = img[pts[0], pts[1]]
        avg = np.average(pixels, axis=0)
        # Lookup nearest BRG color
        dist, idx = kdt.query(avg)
        colors.append(idx)

    # Find relative positions of squares, then place in grid
    cube_face = np.zeros((3,3))
    minX, minY, maxX, maxY = float('inf'), float('inf'), 0, 0    
    for square in squares: 
        center = np.average(square, axis=0)
        minX = min(center[0], minX)
        minY = min(center[1], minY)
        maxX = max(center[0], maxX)
        maxY = max(center[1], maxY)
    for i in range(len(squares)):
        center = np.average(squares[i], axis=0)
        relativeX = round(2*(center[0] - minX) / (maxX - minX))
        relativeY = round(2*(center[1] - minY) / (maxY - minY))
        cube_face[relativeY, relativeX] = colors[i]

    print(cube_face)