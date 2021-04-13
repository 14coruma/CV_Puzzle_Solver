import math
import numpy as np
import cv2 as cv
import imutils
from imutils.perspective import four_point_transform
from sympy import Point, Line, Polygon
from scipy.spatial import ConvexHull
from skimage.segmentation import clear_border

# Euclidean distance between two points
def euclidean(p0,p1): return math.sqrt((p0[0]-p1[0])**2 + (p0[1]-p1[1])**2)

# Given rho and theta value, return two points on the line
# (from https://stackoverflow.com/questions/48954246/find-sudoku-grid-using-opencv-and-python)
def polar_to_points(rho, theta, return_class=False):
    a = math.cos(theta)
    b = math.sin(theta)
    x = a * rho
    y = b * rho
    x0,y0 = int(x + 1000*(-b)), int(y + 1000*(a))
    x1,y1 = int(x - 1000*(-b)), int(y - 1000*(a))
    if return_class: return Point(x0,y0), Point(x1,y1)
    else: return (x0,y0), (x1,y1)

# Given an image, determine location of Sudoku board, and then focus image on board
def locate_puzzle(img):
    # Blurr image to reduce noise in edges
    thresh = cv.GaussianBlur(img, (3,3), cv.BORDER_REFLECT)

    param1, param2 = 15, 20 
    while True:
        # Canny edge detection
        edges = cv.Canny(thresh, param1, param2, None, 3)
        # Dilate and erode edges (from https://stackoverflow.com/questions/48954246/find-sudoku-grid-using-opencv-and-python)
        edges = cv.dilate(edges, np.ones((3,3),np.uint8), iterations=1)
        edges = cv.erode(edges, np.ones((5,5),np.uint8), iterations=1)

        lines = cv.HoughLines(edges, 1, np.pi / 180, 150, None, 0, 0)
        param1, param2 = param1+5, param2+30
        if lines is None: continue
        if len(lines) <= 35: break
        
    # Copy edges to the images that will display the results in BGR
    cdst = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)

    # Find four corners of sudoku board
    poss_corners = []
    for i in range(len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        pt1, pt2 = polar_to_points(rho, theta)
        cv.line(cdst, pt1, pt2, (0,0,255), 3, cv.LINE_AA)
        p1, p2 = Point(pt1), Point(pt2)
        l1 = Line(p1, p2)
        for j in range(len(lines)):
            if i == j: continue
            rho = lines[j][0][0]
            theta = lines[j][0][1]
            p3, p4 = polar_to_points(rho, theta, return_class=True)
            l2 = Line(p3, p4)
            p = l1.intersection(l2)
            if len(p) == 0: continue
            p = np.array([int(p[0][1]), int(p[0][0])])
            if 0 <= p[0] < len(img) and 0 <= p[1] < len(img[1]):
                poss_corners.append(p)
    
    tl = poss_corners[np.argmin(np.array(list(map(euclidean, poss_corners, [[0,0]]*len(poss_corners)))))]
    tr = poss_corners[np.argmin(np.array(list(map(euclidean, poss_corners, [[0,len(img[0])]]*len(poss_corners)))))]
    bl = poss_corners[np.argmin(np.array(list(map(euclidean, poss_corners, [[len(img),0]]*len(poss_corners)))))]
    br = poss_corners[np.argmin(np.array(list(map(euclidean, poss_corners, [[len(img),len(img[0])]]*len(poss_corners)))))]

    # Idea for warping using imutils: https://www.pyimagesearch.com/2020/08/10/opencv-sudoku-solver-and-ocr/
    img = four_point_transform(img, np.array([
        [tl[1],tl[0]],
        [tr[1],tr[0]],
        [bl[1],br[0]],
        [br[1],br[0]]]))
    
    return img

# Given an image of ONLY a sudoku board, determine where the numbers are,
# then use OCR to classify each digit
def construct_board(img):
    height, width = img.shape
    h_cell, w_cell = height//9, width//9
    # Blurr image to reduce noise in edges
    img = cv.GaussianBlur(img, (3,3), cv.BORDER_REFLECT)
    # Threshold
    img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    im2 = cv.bitwise_not(img)
    for i in range(81):
        y, x = w_cell*(i//9), h_cell*(i%9)
        cv.imshow("Hi", img[y:y+h_cell, x:x+w_cell])
        cell = im2[y:y+h_cell, x:x+w_cell]
        cell = clear_border(cell)
        cell = cv.bitwise_not(cell)
        cv.imshow("HI2", cell)
        cv.waitKey()

if __name__ == "__main__":
    img = cv.imread("../Images/sudoku_0_full.png", 0)
    #img = cv.imread("../Images/sudoku.jpg", 0)
    img = locate_puzzle(img)
    board = construct_board(img)