# Code used to solve a Sudoku puzzle, given an image
import numpy as np

NUMS = np.array([1,2,3,4,5,6,7,8,9])

# Check if a sudoku board is valid (ie solved) 
def valid(board):
    for i in range(9):
        # Rows have 1..9
        if not np.array_equal(np.sort(board[i]), NUMS): return False
        # Columns have 1..9
        if not np.array_equal(np.sort(board[:,i]), NUMS): return False
        # Squares have 1..9
        r, c = 3*(i//3), 3*(i%3)
        if not np.array_equal(np.sort(board[r:r+3,c:c+3].flatten()), NUMS): return False
    return True

# Generate list of all legal numbers at a given position
def possible_nums(board, r, c):
    p_nums = np.copy(NUMS)
    for i in range(9):
        # Check numbers already in row
        p_nums = np.delete(p_nums, np.argwhere(p_nums == board[r,i]))
        # Check numbers already in column
        p_nums = np.delete(p_nums, np.argwhere(p_nums == board[i,c]))
        # Check numbers already in square
        sr, sc = 3*(r//3), 3*(c//3)
        p_nums = np.delete(p_nums, np.argwhere(p_nums == board[sr+i//3, sc+i%3]))
    return p_nums

# Given a 9x9 sudoku board, solve recursively using backtracking
def solve(board, r=0, c=0):
    # If at last position, don't recurse further
    if r >= 8 and c >= 8:
        if valid(board): return board
        p_nums = possible_nums(board, r, c)
        for p in p_nums:
            board[r,c] = p
            if valid(board): return board
        return None
    # Check a possible number, then recursively try to see if we can solve
    next_r = r + (c + 1) // 9
    next_c = (c + 1) % 9
    if board[r,c] != 0: return solve(np.copy(board), next_r, next_c)
    p_nums = possible_nums(board, r, c)
    for p in p_nums:
        board[r,c] = p
        result = solve(np.copy(board), next_r, next_c)
        if result is not None: return result
    return None

if __name__ == "__main__":
    board = np.array([
        [0,6,0,0,0,4,0,0,3],
        [0,0,0,3,0,0,0,5,0],
        [3,4,0,0,1,9,0,0,7],
        [0,0,5,0,0,0,4,0,0],
        [7,0,0,8,0,0,0,0,2],
        [0,1,9,0,0,0,0,3,0],
        [0,2,6,0,0,8,0,0,5],
        [8,5,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,9,0]
    ])
    print(solve(board))