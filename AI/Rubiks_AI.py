# Rubiks cube solver thanks to https://github.com/muodov/kociemba
import kociemba

# Given six faces of a rubiks cube, return as a single string
# Faces should be ordered as follows:
# [Front, Right, Back, Left, Upper, Down]
def faceletstr_from_faces(faces):
    # Figure out face names for each color in given cube
    vals = ['F', 'R', 'B', 'L', 'U', 'D']
    facelets = {}
    for i in range(len(faces)):
        facelets[faces[i][1][1]] = vals[i]
    # Create facelet string, using proper ordering expected by kociemba
    # [Upper, Right, Front, Down, Left, Back]
    faceletstr = ""
    for faces_idx in [4, 1, 0, 5, 3, 2]:
        for row in faces[faces_idx]:
            for color in row:
                faceletstr += facelets[color]
    return faceletstr

def solve(faceletstr):
    return kociemba.solve(faceletstr)

if __name__ == "__main__":
    print(faceletstr_from_faces(
        [[['y','o','o'],
        ['o','y','y'],
        ['r','b','b']]]
    ))