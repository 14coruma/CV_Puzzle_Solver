# Code used to solve an Akari puzzle, given a parsed board
import numpy as np

# General board notation:
# 10x10 grid
# Numbers 0-4 represent number of adjacent lights
# Number -1 represents an empty space
# Number 9 represents a wall
# Number 8 represents a light
# Number -2 represents a marked lit space

# Check if an akari board is valid/solved
def valid(board):
    # Check that lights and adjacent numbers match up
    for y in range(10):
        for x in range(10):
            if board[y,x] in [-1, 8, 9]: continue
            light_count = 0
            if y>0 and board[y-1,x] == 8: light_count += 1
            if y<9 and board[y+1,x] == 8: light_count += 1
            if x>0 and board[y,x-1] == 8: light_count += 1
            if x<9 and board[y,x+1] == 8: light_count += 1
            if light_count != board[y,x]:
                print(y, x)
                return False
    # Check that lights do not shine on each other
    # Check that all spaces are lit, by marking them with -2
    lights = np.where(board == 8)
    for y,x in zip(lights[0], lights[1]):
        # Shine up
        for i in reversed(range(y)):
            if board[i,x] == 8: return False
            elif board[i,x] > 0: break
            else: board[i,x] = -2
        # Shine down
        for i in range(y+1,10):
            if board[i,x] == 8: return False
            elif board[i,x] > 0: break
            else: board[i,x] = -2
        # Shine right
        for i in range(x+1,10):
            if board[y,i] == 8: return False
            elif board[y,i] > 0: break
            else: board[y,i] = -2
        # Shine left
        for i in reversed(range(x)):
            if board[y,i] == 8: return False
            elif board[y,i] > 0: break
            else: board[y,i] = -2
    return True

if __name__ == "__main__":
    board = np.array([
        [-1, -1, -1, -1, -1, -1, -1,  0,  9,  9],
        [-1,  9,  2,  9, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1,  9, -1,  9, -1, -1],
        [-1,  1,  9,  9, -1,  9, -1,  1, -1, -1],
        [-1, -1, -1, -1, -1,  3, -1,  9, -1, -1],
        [-1, -1,  1, -1,  3, -1, -1, -1, -1, -1],
        [-1, -1,  9, -1,  9, -1,  9,  1,  9, -1],
        [-1, -1,  1, -1,  9, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1,  9,  9,  2, -1],
        [ 9,  9,  1, -1, -1, -1, -1, -1, -1, -1]
    ])
    #board = np.array([
    #    [1,0,-1,8,1,-1,9,-1,8,1],
    #    [8,-1,-1,-1,2,8,2,8,-1,1],
    #    [-1,8,-1,1,8,-1,-1,9,-1,8],
    #    [-1,2,8,9,-1,9,-1,9,-1,9],
    #    [-1,9,-1,-1,-1,1,8,-1,-1,9],
    #    [9,-1,-1,-1,9,-1,-1,-1,9,-1],
    #    [9,-1,1,8,2,8,9,-1,1,8],
    #    [8,-1,9,-1,-1,-1,9,-1,-1,-1],
    #    [2,8,-1,9,-1,0,-1,-1,8,-1],
    #    [9,-1,8,2,8,1,-1,8,2,9]
    #])
    print("Input board:\n", board)
    print("Solution:\n")
    print(valid(board))