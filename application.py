import pygame
from pygame.locals import DOUBLEBUF,OPENGL 
from OpenGL.GL import *
from OpenGL.GLU import *

from objdata import read_mesh 

import numpy as np
import numpy.linalg as la

from projective_math import *
from quaternion import Quaternion

mesh_files = [
    'assets/rabbit.obj',    # Slow
    'assets/teapot.obj',    # Slow
    'assets/cat.obj',       # Slow 
    'assets/bunny.off',     # Slow 
    'assets/torus.obj',     # Fast
    'assets/cinda.obj',     # Fast
    'assets/samoyed.obj',   # Fast
    'assets/suzanne.obj',   # Fast
]

mesh_file = mesh_files[-1]
vertices, faces, normals = read_mesh(mesh_file)

class App:
    def on_create(self):
        self.translation = np.array([0.0, 0.0,  0.0], dtype=float) 
        self.perspective = np.array([0.0, 0.05, 1.2], dtype=float) 
        self.scale:float = 3.5

        self.angle:float = 1.2
        self.axis = np.array([-1, 0.5, 1],dtype=float)

        self.light_direction = np.array([0.0, 2.0, 1.0], dtype=float)
        self.light_rotation  = rotatex(0.15)

        self.rotate = True 
        # It's faster using matrix, because it leverages the numpy library
        # But I left this here as a demonstrations that we can just multiply 
        # the quaternions and vector as usual, no problem i.e -> q v q*
        self.use_quaternion_matrix = True 


        clear_color = np.array([255,255,255,255])/255.0
        glClearColor(*clear_color) 
        glEnable(GL_DEPTH_TEST)

        self.is_down = {
            pygame.K_w : False,
            pygame.K_s : False,
            pygame.K_a : False,
            pygame.K_d : False,
            pygame.K_e : False,
            pygame.K_q : False,
            pygame.K_z : False,
            pygame.K_x : False,
        }

    def on_event(self,e) : 
        if e.type == pygame.QUIT:
            pygame.quit()
            exit()
        
        
        for key in self.is_down.keys():
            if e.type == pygame.KEYUP:
                if e.key == key:
                    self.is_down[key] = False
            if e.type == pygame.KEYDOWN:
                if e.key == key:
                    self.is_down[key] = True 

        if e.type == pygame.KEYDOWN:
            if e.key == pygame.K_ESCAPE:
                pygame.quit()
                exit()
            if e.key == pygame.K_SPACE:
                self.rotate = not self.rotate

        if e.type == pygame.MOUSEWHEEL:
            if   e.y < 0:
                self.scale += 0.15 
            elif e.y > 0:
                self.scale -= 0.15

    def on_update(self):
        is_down = self.is_down

        if self.rotate:
            self.angle += 0.025

        if is_down[pygame.K_w]:
            self.translation[1] += 0.1
        if is_down[pygame.K_s]:
            self.translation[1] -= 0.1

        if is_down[pygame.K_d]:
            self.translation[0] += 0.1
        if is_down[pygame.K_a]:
            self.translation[0] -= 0.1

        if is_down[pygame.K_e]:
            self.light_direction = (np.array([*self.light_direction,1.0], dtype=float) @ self.light_rotation)[:-1] 
        if is_down[pygame.K_q]:
            self.light_direction = (np.array([*self.light_direction,1.0], dtype=float) @ self.light_rotation.T)[:-1] 

        if is_down[pygame.K_z]:
            self.perspective[2] += 0.035
        if is_down[pygame.K_x]:
            self.perspective[2] -= 0.035

        q = Quaternion.from_axis_angle(axis=self.axis, angle=self.angle)

        M = projective_transform(
            linear=np.eye(3),
            scale=self.scale,
            translation= self.translation,
            perspective= self.perspective
        )
        if self.use_quaternion_matrix:
            rotation = q.to_matrix()
            M = M @ rotation
 
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        #glMatrixMode(GL_PROJECTION)
        #glLoadIdentity()
        #glMatrixMode(GL_MODELVIEW)
        #glLoadIdentity()

        glBegin(GL_TRIANGLES)
        for index,face in enumerate(faces):
            v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]

            if not self.use_quaternion_matrix:
                v0 = q.rotate(v0)
                v1 = q.rotate(v1)
                v2 = q.rotate(v2)


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
            light_direction = self.light_direction/la.norm(self.light_direction)
            color = (255,123,123)
            normal = normals[index]
            intensity = np.dot(normal, light_direction)
            color = np.abs(intensity) * np.array(color)/256.0
            glColor3f(*color)
            glVertex3f(*v0)
            glVertex3f(*v1)
            glVertex3f(*v2)
        glEnd()


def main():
    pygame.init()
    window_size = (600, 600)
    pygame.display.set_mode(window_size, DOUBLEBUF | OPENGL)

    app = App()
    app.on_create()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            app.on_event(event)
        app.on_update()
        pygame.display.flip()

if __name__ == '__main__':
    main()