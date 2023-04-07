import numpy as np

# >> Rotations 
def rotatez(theta: float):
    cos = np.cos(theta)
    sin = np.sin(theta)
    return np.array([
        [cos, -sin, 0, 0],
        [sin,  cos, 0, 0],
        [0,    0,   1, 0],
        [0,    0,   0, 1],
    ],dtype=float)


def rotatey(theta: float):
    cos = np.cos(theta)
    sin = np.sin(theta)
    return np.array([
        [ cos, 0, sin, 0 ],
        [ 0,   1, 0,   0 ],
        [-sin, 0, cos, 0 ],
        [ 0,   0, 0,   1 ],
    ],dtype=float)

def rotatex(theta: float):
    cos = np.cos(theta)
    sin = np.sin(theta)
    return np.array([
        [ 1, 0,    0,   0 ],
        [ 0, cos, -sin, 0 ],
        [ 0, sin,  cos, 0 ],
        [ 0, 0,      0, 1 ],
    ],dtype=float)
# <<

# >> Projetions

def projective_transform(linear,translation, perspective, scale:int) -> np.ndarray:
    L = np.array(linear)
    T = np.array(translation)
    P = np.array(perspective)
    return np.array([
        [ L[0,0], L[0,1], L[0,2], T[0] ],
        [ L[1,0], L[1,1], L[1,2], T[1] ],
        [ L[2,0], L[2,1], L[2,2], T[2] ],
        [ P[0],   P[1],   P[2],  scale ]
    ],dtype=float)



def project_isometry(point, v) -> np.ndarray:
    p = point
    t = -p[2]/v[2]
    return np.array([ p[0]+t*v[0], p[1]+t*v[1] ])


def project_perspective(point, eye) -> np.ndarray: 
    p = point - np.array(eye).reshape(point.shape)
    z = p[2]
    return np.array([p[0]/z, p[1]/z])
# << 

def quaternion2matrix():
    pass