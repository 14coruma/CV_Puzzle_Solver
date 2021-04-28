import math
import numpy as np
import sys

import cv2 as cv
from scipy.spatial import KDTree

def normalize(img, debug=False):
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    l,a,b = cv.split(lab)
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv.merge((cl,a,b))
    normalized = cv.cvtColor(limg, cv.COLOR_LAB2BGR)
    if debug:
        cv.imshow("color", normalized)
        cv.waitKey()
        cv.destroyWindow("color")
    return normalized

def get_squares(img, debug=False):
    normalized = normalize(img)

    # Threshold image for very colorful or white pixels (like you'd expect on a rubik's cube)
    blurred = cv.GaussianBlur(normalized, (5,5), cv.BORDER_REFLECT)
    hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)
    #color = cv.inRange(hsv, (0,100,100), (180,255,255)) # HSV Colors (any H, high S, high V)
    color = cv.inRange(hsv, (0,75,100), (180,255,255)) # HSV Colors (any H, high S, high V)
    white = cv.inRange(hsv, (0,0,200), (180,80,255))    # HSV White (any H, low S, very high V)
    thresh = cv.bitwise_or(color, white)
    #thresh = cv.dilate(thresh, np.ones((3,3),np.uint8), iterations=1)
    if debug:
        cv.imshow("color", thresh)
        cv.waitKey()
        cv.destroyWindow("color")

    # Edge detection
    blurred = cv.GaussianBlur(thresh, (3,3), cv.BORDER_REFLECT)
    edges = cv.Canny(blurred, 25, 250)
    if debug:
        cv.imshow("color", edges)
        cv.waitKey()
        cv.destroyWindow("color")

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
        # Also need at least 100 pixels to consider a box (obviously this is dependant on a webcam with decent resolution)
        if cnt_area > .9 * box_area and box_area > 100:
            squares.append(box)
    squares = sorted(squares, key=cv.contourArea)[-9:]
    if debug:
        img_copy = img.copy()
        cv.drawContours(img_copy, squares, -1, (0,0,255), 2)
        cv.imshow("squares", img_copy)
        cv.waitKey()
        cv.destroyWindow("squares")

    return squares

def get_colors(img, squares):
    # Identify color of each square
    # Idea to use KDTree for nearest color lookup: https://medium.com/codex/rgb-to-color-names-in-python-the-robust-way-ec4a9d97a01f
    # List of possible rubik's cube BGR colors:
    bgr_names = ['red', 'green', 'blue', 'yellow', 'orange', 'white']
    bgr_colors = [
        [20,20,235],  # Red 
        [20,235,20],  # Green
        [235,20,20],  # Blue
        [20,215,215], # Yellow
        [0,145,235], # Orange
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
        colors.append(bgr_names[idx])
        if bgr_names[idx] == 'orange': print(avg)

    return colors

def label_face(squares, colors):
    # Find relative positions of squares, then place in grid
    cube_face = np.empty((3,3), dtype=str)
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
    return cube_face

def face_from_image(img, debug=False):
    squares = get_squares(img, debug)
    colors = get_colors(img, squares)
    cube_face = label_face(squares, colors)
    return cube_face

def face_from_filename(filename):
    img = cv.imread(filename)
    return face_from_image(img)

def faces_from_webcam():
    # Video Capture code taken from https://stackoverflow.com/questions/604749/how-do-i-access-my-webcam-in-python
    cv.namedWindow("Webcam")
    vc = cv.VideoCapture(0)
    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False
    
    faces = []
    face_names = ['Front', 'Right', 'Back', 'Left', 'Upper', 'Down']
    # Loop through each face name, getting extracting rubik's data from webcam frame
    for face_name in face_names:
        while rval:
            # Show squares, so user knows if properly being detected
            squares = get_squares(frame)
            cv.drawContours(frame, squares, -1, (0,0,255), 2)
            # Tell user what to do...
            cv.putText(
                frame, "Press [ENTER] once {} is pictured...".format(face_name),
                (8, 32), cv.FONT_HERSHEY_SIMPLEX, .9, (255,255,255), 2, cv.LINE_AA)
            # Show webcam
            cv.imshow("Webcam", frame)
            rval, frame = vc.read()
            key = cv.waitKey(40)
            if key == ord('\n') or key == ord('\r'): break
        faces.append(face_from_image(frame, debug=True))

    cv.destroyWindow("Webcam")
    return faces

if __name__ == "__main__":
    debug = True

    rubiks_image_folder = "Images/rubiks_0/"
    filename = "Images/rubiks_0/back.jpg"
    img = cv.imread(filename)

    squares = get_squares(img, debug=debug)
    colors = get_colors(img, squares)
    cube_face = label_face(squares, colors)

    print(cube_face)

    #faces = faces_from_webcam()
    #for face in faces:
    #    print(face)