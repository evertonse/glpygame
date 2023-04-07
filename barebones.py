import pygame
import sys
from pygame.locals import DOUBLEBUF,OPENGL 
from OpenGL.GL import *
from OpenGL.GLU import *

from objdata import ObjData
import numpy as np
from projective_math import *
from pathlib import Path 

# Initialize Pygame
pygame.init()
# Set up Pygame window
window_size = (800, 600)
pygame.display.set_mode(window_size, DOUBLEBUF | OPENGL)

def read_mesh(filepath, normalize=False):
    ext = Path(filepath).suffix.lower()
    if ext in {'.off', 'off'}:
        obj = ObjData.from_off(filepath)
    elif ext in {'.obj', 'obj'}:
        obj = ObjData.from_obj(filepath)
    else:
        raise Exception("We Only support .obj or .off files")

    vertices = np.array(obj.vertices,dtype=float)
    faces = np.array(obj.faces,dtype=int)  

    normals = []
    for face in faces:
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        normal = np.cross(v2 - v0, v1 - v0)
        normal /= np.linalg.norm(normal)
        normals.append(normal)
    
    normals = np.array(normals,dtype=float)  
    abs_max = np.abs(vertices.max())
    abs_min = np.abs(vertices.min())
    max_val = max(abs_max, abs_min)
    return vertices/max_val, faces, normals


file = 'assets/bunny.off'
file = 'assets/cat.obj'
file = 'assets/teapot.obj'
file = 'assets/rabbit.obj'
file = 'assets/torus.obj'
file = 'assets/suzanne.obj'
vertices, faces, normals = read_mesh(file)
glClearColor(0.2, 0.3, 0.3, 1.0) 
glEnable(GL_DEPTH_TEST)

translation = projective_transform(
    linear=np.eye(3),
    scale=1,
    translation=[0,0,0],
    perspective=[0,0,0],
)

perspective = projective_transform(
    linear=np.eye(3),
    scale=2.5,
    translation=[0,0,0],
    perspective=[0,0.1,0.5],
)

angle = 2


light_direction = np.array([0.0, 2.0, 1.0], dtype=float)
light_direction /= np.linalg.norm(light_direction)

while True:
    angle+=0.1
    M = translation @ perspective  @ rotatez(angle) @ rotatex(angle)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
    # Clear the screen
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()

    #glOrtho(
    #    # planos near far, left right, top bottom, e pá
    #    -10.0, 10.0, 
    #    -10.0, 10.0,
    #    -10.0, 10.0
    #)

    #gluLookAt(0, 0, 5, 0, 0, 0, 0, 1, 0)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()


    glBegin(GL_TRIANGLES)
    for index,face in enumerate(faces):
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]

        v0 = M @ np.array([[*v0,1.0]]).T
        v1 = M @ np.array([[*v1,1.0]]).T
        v2 = M @ np.array([[*v2,1.0]]).T

        # Perspective Divide
        v0 /= v0[-1]
        v1 /= v1[-1]
        v2 /= v2[-1]

        v0 = v0[:-1]
        v1 = v1[:-1]
        v2 = v2[:-1]

        color = (255,123,123)
        normal = normals[index]
        intensity = np.dot(normal, light_direction)
        color = np.abs(intensity) * np.array(color)/256.0
        glColor3f(*color)
        glVertex3f(*v0)
        glVertex3f(*v1)
        glVertex3f(*v2)
    glEnd()

    # Update the screen
    pygame.display.flip()