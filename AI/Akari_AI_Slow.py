# Code used to solve an Akari puzzle, given a parsed board
import numpy as np

SIZE = 4
COUNT = 0

# General board notation:
# SIZExSIZE grid
# Numbers 0-4 represent number of adjacent lights
# Number -1 represents an empty space
# Number 9 represents a wall
# Number 8 represents a light
# Number -2 represents a marked lit space

# Check if an akari board is valid/solved
def valid(board):
    # Check that lights and adjacent numbers match up
    for y in range(SIZE):
        for x in range(SIZE):
            if board[y,x] in [-2, -1, 8, 9]: continue
            light_count = 0
            if y>0 and board[y-1,x] == 8: light_count += 1
            if y<SIZE-1 and board[y+1,x] == 8: light_count += 1
            if x>0 and board[y,x-1] == 8: light_count += 1
            if x<SIZE-1 and board[y,x+1] == 8: light_count += 1
            if light_count != board[y,x]: return False
    # Check that lights do not shine on each other
    # Mark lit spaces with -2
    lights = np.where(board == 8)
    for y,x in zip(lights[0], lights[1]):
        # Shine up
        for i in reversed(range(y)):
            if board[i,x] == 8: return False
            elif board[i,x] >= 0: break
            else: board[i,x] = -2
        # Shine down
        for i in range(y+1,SIZE):
            if board[i,x] == 8: return False
            elif board[i,x] >= 0: break
            else: board[i,x] = -2
        # Shine right
        for i in range(x+1,SIZE):
            if board[y,i] == 8: return False
            elif board[y,i] >= 0: break
            else: board[y,i] = -2
        # Shine left
        for i in reversed(range(x)):
            if board[y,i] == 8: return False
            elif board[y,i] >= 0: break
            else: board[y,i] = -2
    # Check that all spaces are lit
    for y in range(SIZE):
        for x in range(SIZE):
            if board[y,x] == -1: return False
    return True

# Mark lit spaces if light at position y,x
# If lighting another light, then return None
def mark_lights(board, y, x):
    # Shine up
    for i in reversed(range(y)):
        if board[i,x] == 8: return None
        elif board[i,x] >= 0: break
        else: board[i,x] = -2
    # Shine down
    for i in range(y+1,SIZE):
        if board[i,x] == 8: return None
        elif board[i,x] >= 0: break
        else: board[i,x] = -2
    # Shine right
    for i in range(x+1,SIZE):
        if board[y,i] == 8: return None
        elif board[y,i] >= 0: break
        else: board[y,i] = -2
    # Shine left
    for i in reversed(range(x)):
        if board[y,i] == 8: return None
        elif board[y,i] >= 0: break
        else: board[y,i] = -2
    board[y,x] = 8
    return board

# Given a SIZExSIZE akari board, solve recursively using backtracking
def solve(board, r=0, c=0):
    # If at last position, don't recurse further
    if r >= SIZE-1 and c >= SIZE-1:
        if valid(board): return board
        if board[r,c] == -1: board = mark_lights(board, r, c)
        if board is not None and valid(board): return board
        global COUNT
        COUNT += 1
        return None
    
    next_r = r + (c+1) // SIZE
    next_c = (c+1) % SIZE
    # If not an empty, unlit space, move to next position
    if board[r,c] != -1: return solve(np.copy(board), next_r, next_c)
    # Check with a light at current position, if possible
    board2 = mark_lights(np.copy(board), r, c)
    if board2 is not None:
        board2[r,c] = 8
        result = solve(np.copy(board2), next_r, next_c)
        if result is not None: return result
    # Check leaving current space empty
    result = solve(np.copy(board), next_r, next_c)
    if result is not None: return result
    # Otherwise, no solution possible...
    return None

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
    board = board[:SIZE,:SIZE]
    print("Input board:\n", board)
    print("Solution:\n", solve(board))
    print(COUNT)