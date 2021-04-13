import math
import numpy as np
import sys

import cv2 as cv
import imutils
from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
from sympy import Point, Line

from tensorflow.keras import models

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
def locate_puzzle(img, debug=False):
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

    if debug: 
        cv.imshow("Lines", cdst)
        cv.imshow("Cropped", img)
        cv.waitKey()
    
    return img

def center_by_mass(cell, debug=False):
    center = [0,0]
    for i in range(len(cell)):
        for j in range(len(cell[0])):
            center[0] += i * cell[i,j]
            center[1] += j * cell[i,j]
    center = center / np.sum(cell)
    if debug:
        cv.imshow("Cell", cell)
        cv.waitKey()
    M = np.float32([
        [1,0, int((cell.shape[1]-1)/2-center[1])],
        [0,1, int((cell.shape[0]-1)/2-center[0])]])
    cell = cv.warpAffine(cell, M, cell.shape)
    return cell

def get_digit(cell, ocr_model, debug=False):
    # BEGIN: Code adapted from https://www.pyimagesearch.com/2020/08/10/opencv-sudoku-solver-and-ocr/
    cell = cv.threshold(cell, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
    cell = clear_border(cell)
    cell = clear_border(cell)
    # compute the percentage of area of thresholded pixels
    (h, w) = cell.shape
    percentFilled = np.count_nonzero(cell == 255) / float(w * h)
    # if less than 3% of the mask is filled, then must be empty cell
    if percentFilled < 0.03: return 0
    # END: Code adapted from https://www.pyimagesearch.com/2020/08/10/opencv-sudoku-solver-and-ocr/
    # Resize to closer match MNIST dataset
    cell = center_by_mass(cell)
    cell = cv.resize(cell, (36,36))
    # Focus in on center of cell just a little, but make sure final size is (28,28)
    cell = four_point_transform(cell, np.array([[4,4],[4,32],[32,4],[32,32]]))
    cell = center_by_mass(cell)
    # Use pre-trained OCR_CNN (on MNIST dataset) to classify number
    digit = ocr_model.predict(np.array([cell]).reshape(1,28,28,1) / 255.0)
    # label is returned as a one-hot categorical array. Need to cast to integer
    digit = np.argmax(digit, axis=-1) 

    if debug:
        print("Digit: ", digit)
        cv.imshow("Cell", cell)
        cv.waitKey()

    return digit

# Given an image of ONLY a sudoku board, determine where the numbers are,
# then use OCR to classify each digit
def construct_board(img, ocr_model, debug=False):
    board = np.zeros((9,9))
    height, width = img.shape
    h_cell, w_cell = height//9, width//9
    # Blurr image to reduce noise in edges
    img = cv.GaussianBlur(img, (3,3), cv.BORDER_REFLECT)
    for i in range(81):
        y, x = h_cell*(i//9), w_cell*(i%9)
        cell = img[y:y+h_cell, x:x+w_cell]
        board[i//9, i%9] = get_digit(cell, ocr_model, debug)
    return board

# Given an image of a cropped Sudoku board, and the parsed board[][],
# display the digits of the board[][] on the image
def visualize_board(img, board):
    height, width = img.shape
    h_cell, w_cell = height//9, width//9
    img = cv.cvtColor(img, cv.COLOR_GRAY2RGB) # Let's show the image in color
    for i in range(81):
        y, x = h_cell*(i//9+1), w_cell*(i%9)
        digit = int(board[i//9][i%9])
        if digit == 0: digit = ""
        # BEGIN: Code adapted from https://www.pyimagesearch.com/2020/08/10/opencv-sudoku-solver-and-ocr/
        textX = int(w_cell * 0.4)
        textY = int(h_cell * -0.3)
        textX += x
        textY += y
        cv.putText(img, str(digit), (textX, textY), cv.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)
        # END: Code adapted from https://www.pyimagesearch.com/2020/08/10/opencv-sudoku-solver-and-ocr/
    return img

if __name__ == "__main__":
    filename, debug = "Images/sudoku.jpg", False
    if len(sys.argv) > 1: filename = sys.argv[1]
    img = cv.imread(filename, 0)
    ocr_model = models.load_model('Models/OCR_CNN_Trained')
    cropped = locate_puzzle(img, debug)
    board = construct_board(cropped, ocr_model, debug)
    print(board)
    visualized = visualize_board(cropped, board)
    cv.imshow("Visualized", visualized)
    cv.waitKey()