from enum import Enum
from functools import partial
import pygame
from objdata import ObjData
import numpy as np
import numpy.linalg as la

from math import *

RELEASE = True
RELEASE = False

if RELEASE:
    print = lambda *x:x

pygame.display.set_caption("Isometry projection in pygame! Brought to you by ExCyber")

# >> ###### CONSTANTES #####
WHITE = (255,   255, 255)
RED   = (255,   0,   0  )
BLACK = (0,     0,   0  )

WIDTH, HEIGHT   = 1600, 900
ASPECT_RATIO    = float(WIDTH)/HEIGHT

SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))

TRANSLATION = np.array([WIDTH/2,HEIGHT/2])
# <<

def ObjFile(filename):
    obj = ObjData.from_obj(filename)
    edges = []
    for f in obj.faces:
        edges.append([f[0],f[1]])
        edges.append([f[1],f[2]])
        edges.append([f[2],f[0]])
    edges = np.array(edges,dtype=int)
    vertices = np.array(obj.vertices,dtype=float)
    return vertices, edges 


def OffFile(filename, make_edges=True,make_normals=False):
    obj = ObjData.from_off(filename)
    vertices = np.array(obj.vertices,dtype=float)

    if make_edges:
        edges = []
        for f in obj.faces:
            edges.append([f[0],f[1]])
            edges.append([f[1],f[2]])
            edges.append([f[2],f[0]])
        edges = np.array(edges,dtype=int)
        return vertices, edges 

    faces = np.array(obj.faces,dtype=int)  
    normals = []
    if make_normals:
        for face in faces:
            v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
            normal = np.cross(v1 - v0, v2 - v0)
            normal /= np.linalg.norm(normal)
            normals.append(normal)
        
        normals = np.array(normals,dtype=float)  
        return vertices, faces, normals
    return vertices, faces


# >> PRIMITIVES
def Cube() -> tuple:
    vertices = np.array([
        [1,0,0], # A 0
        [1,1,0], # B 1
        [0,1,0], # C 2
        [0,0,0], # D 3 
        [1,0,1], # E 4
        [1,1,1], # F 5
        [0,1,1], # G 6
        [0,0,1], # H 7
    ],dtype=float)

    edges = np.array([
        [0,4], # A-E
        [4,5], # E-F
        [5,1], # F-B
        [5,6], # F-G
        [1,0], # B-A
        [0,3], # A-D
        [3,2], # D-C
        [2,1], # C-B
        [2,6], # C-G
        [7,6], # H-G
        [4,7], # E-H
        [7,3], # H-D
        
    ],dtype=int)

    return vertices,edges
# << 

# >> Rotations 
def rotatez(theta: float):
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1],
    ])


def rotatey(theta: float):
    return np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)],
    ])

def rotatex(theta: float):
    return np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)],
    ])
# <<

# >> Projetions
def project_isometry(point, v) -> np.ndarray:
    p = point
    t = -p[2]/v[2]
    return np.array([ p[0]+t*v[0], p[1]+t*v[1] ])


def project_perspective(point, eye) -> np.ndarray: 
    p = point - np.array(eye).reshape(point.shape)
    z = p[2]
    return np.array([p[0]/z, p[1]/z])
# << 


# >> DRAW
def draw_triangle(point,color):
    pygame_color = pygame.Color(color)
    pygame.draw.polygon(SCREEN, pygame_color , (point[0],point[1],point[2]))


def draw_triangles(points,faces,normals=None, light_direction = None):
    P = np.array(points,dtype=int)
    for index,(i,j,k) in enumerate(faces):
        point = (P[i],P[j],P[k])
        color = (255,123,123)
        if light_direction is not None:
            normal = normals[index]
            intensity = abs(np.dot(normal, light_direction))
            #or np.clip(intensity, 0, 1) 
            color = tuple(intensity * np.array(color))  # Use intensity to determine color
        draw_triangle(point, color)

def draw_edges(points,edges):
  P = np.array(points,dtype=int)
  for i, j in edges:
    print(i,j)
    p1,p2 = points[i],points[j] 
    pygame.draw.aaline(
        SCREEN, BLACK, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])))
# <<



class ViewMode(Enum):
    PERSPECTIVE = 0,
    ISOMETRIC   =  1,

class App:
    def __init__(self) -> None:
        self.vertices = None
        self.edges = None 
        self.faces = None 
        self.normals = None 
        self.angle: float   = 0.0
        self.view_vector    = np.array([-1.0, 1.0,  -550.0])
        self.angle_dt       = 0.01
        self.view_dt: float = 0.0
        self.rotate: bool = True
        self.eye = np.array([0.1,0.1,6.0], dtype=float)
        self.wire_frame = False
        self.continues_down = {
            pygame.K_w : False,
            pygame.K_s : False,
            pygame.K_a : False,
            pygame.K_d : False,
        }

        self.light_direction = np.array([1.0, 1.0, 1.0], dtype=float)
        self.light_direction /= la.norm(self.light_direction)

    def on_create(self):
        objfiles_options = [
            'assets/cat.obj',
            'assets/rabbit.obj',
        ]
        offfiles_options = [
            'assets/bunny.off'
        ]

        #vertices, edges = Cube()
        #vertices, edges = ObjFile(objfiles_options[0])
        if self.wire_frame:
            vertices, edges = OffFile(offfiles_options[0])
            self.edges = edges
        else:
            vertices, faces,normals = OffFile(offfiles_options[0], make_edges=False,make_normals=True)
            self.faces = faces
            self.normals = normals

        abs_max_val = np.abs(vertices.max())
        abs_min_val = np.abs(vertices.min())
        max_val = abs_max_val if abs_max_val > abs_min_val else abs_min_val
        self.vertices = vertices/max_val
            

    # >> Displaying
    def on_event(self,e) : 
        if e.type == pygame.QUIT:
            pygame.quit()
            exit()

        if e.type == pygame.KEYUP:
            for key in {pygame.K_w, pygame.K_s, pygame.K_a, pygame.K_d}:
                self.continues_down[key] = False if bool(e.key == key) else self.continues_down[key]
        
        if e.type == pygame.KEYDOWN:
            for key in {pygame.K_w, pygame.K_s, pygame.K_a, pygame.K_d}:
                self.continues_down[key] = True if bool(e.key == key) else self.continues_down[key]

            if e.key == pygame.K_ESCAPE:
                pygame.quit()
                exit()
            if e.key == pygame.K_SPACE:
                self.rotate = not self.rotate

    def on_update(self, ):
        if self.rotate:
            self.angle       += self.angle_dt 
            self.view_vector += self.view_dt 
        key_w, key_s, key_a, key_d = pygame.K_w, pygame.K_s, pygame.K_a, pygame.K_d
        
        if self.continues_down[key_w] ==  True:
            self.eye[2] -= 0.2
        if self.continues_down[key_s] ==  True:
            self.eye[2] += 0.2
        if self.continues_down[key_a] ==  True:
            self.eye[1] -= 0.1
        if self.continues_down[key_d] ==  True:
            self.eye[1] += 0.1

        angle = self.angle
        view_vector = self.view_vector
        vertices,edges,faces,normals = self.vertices, self.edges,self.faces,self.normals

        MODE =  ViewMode.ISOMETRIC
        MODE =  ViewMode.PERSPECTIVE
        
        if MODE == ViewMode.ISOMETRIC: 
            scale = 100 
            v = np.array(view_vector, dtype=float)
            project = partial(project_isometry,v=v)


        elif MODE == ViewMode.PERSPECTIVE:
            scale =  5000
            eye = self.eye
            project = partial(project_perspective,eye=eye)

        points = list()
        for point in vertices:
            p = point
            rotation = rotatez(angle) @ rotatey(angle) @ rotatex(angle)
            p = rotation @ point.reshape((3, 1))
            p = (project(p).flatten())*scale +TRANSLATION
            
            points.append(p)

        if self.wire_frame:
            draw_edges(points,edges)
        else:
            draw_triangles(points,faces,normals=normals,light_direction=self.light_direction)
# <<

# Start
def main():
    app = App()
    clock = pygame.time.Clock()
    app.on_create()
    while True:
        clock.tick(60) # I supposed this means 60 fps
        SCREEN.fill(WHITE)

        for event in pygame.event.get():
            app.on_event(event)

        app.on_update()
        pygame.display.update()


if __name__ == '__main__':
    main()