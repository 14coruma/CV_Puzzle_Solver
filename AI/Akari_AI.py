import constraint
import numpy as np

SIZE=10

def solve(board):
    problem = constraint.Problem(constraint.BacktrackingSolver())
    
    # Build Variables
    for y in range(SIZE):
        for x in range(SIZE):
            if board[y,x] == -1:
                problem.addVariable(str([y,x]), [1, 100])

    # Constraints for variables near numbers
    for y in range(SIZE):
        for x in range(SIZE):
            if board[y,x] in [0,1,2,3,4]:
                val = 0
                adj_vars = []
                if y>0 and board[y-1,x] == -1: adj_vars.append(str([y-1,x]))
                if y<SIZE-1 and board[y+1,x] == -1: adj_vars.append(str([y+1,x]))
                if x>0 and board[y,x-1] == -1: adj_vars.append(str([y,x-1]))
                if x<SIZE-1 and board[y,x+1] == -1: adj_vars.append(str([y,x+1]))
                val += 100*board[y,x] + len(adj_vars) - board[y,x]
                problem.addConstraint(constraint.ExactSumConstraint(val), adj_vars)

    # Sum of each scan must be less than 200, since only 1 light per scan
    # For horizontal scans 
    for y in range(SIZE):
        curr_scan = []
        for x in range(SIZE):
            if board[y,x] == -1: curr_scan.append(str([y,x]))
            else:
                if len(curr_scan) > 0:
                    problem.addConstraint(constraint.MaxSumConstraint(199), curr_scan)
                curr_scan = []
        if len(curr_scan) > 0:
            problem.addConstraint(constraint.MaxSumConstraint(199), curr_scan)
    # For vertical scans 
    for x in range(SIZE):
        curr_scan = []
        for y in range(SIZE):
            if board[y,x] == -1: curr_scan.append(str([y,x]))
            else:
                if len(curr_scan) > 0:
                    problem.addConstraint(constraint.MaxSumConstraint(199), curr_scan)
                curr_scan = []
        if len(curr_scan) > 0:
            problem.addConstraint(constraint.MaxSumConstraint(199), curr_scan)

    # For each point, pair of scans passing through point must contain 100 at least once
    scans = []
    for y in range(SIZE):
        for x in range(SIZE):
            if board[y,x] != -1: continue
            scan = []
            scan.append(str([y,x]))
            # Up
            for i in reversed(range(y)):
                if board[i,x] >= 0: break
                else: scan.append(str([i,x]))
            # Down
            for i in range(y+1,SIZE):
                if board[i,x] >= 0: break
                else: scan.append(str([i,x]))
            # Left
            for i in reversed(range(x)):
                if board[y,i] >= 0: break
                else: scan.append(str([y,i]))
            # Right
            for i in range(x+1,SIZE):
                if board[y,i] >= 0: break
                else: scan.append(str([y,i]))
            scan.sort()
            if scan not in scans: scans.append(scan)
    for scan in scans:
        problem.addConstraint(constraint.SomeInSetConstraint([100]), scan)
    
    solution = problem.getSolution()

    for y in range(SIZE):
        for x in range(SIZE):
            if str([y,x]) in solution:
                board[y,x] = -1 if solution[str([y,x])] == 1 else 100
    return board

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

    res = solve(board)
    print(res)