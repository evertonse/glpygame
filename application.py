import pygame
from pygame.locals import DOUBLEBUF,OPENGL 
from OpenGL.GL import *
from OpenGL.GLU import *

from objdata import read_mesh 

import numpy as np
import numpy.linalg as la

from projective_math import *
from quaternion import Quaternion

# Initialize Pygame
pygame.init()
# Set up Pygame window
window_size = (800, 600)
pygame.display.set_mode(window_size, DOUBLEBUF | OPENGL)



file = 'assets/bunny.off'
file = 'assets/cat.obj'
file = 'assets/teapot.obj'
file = 'assets/rabbit.obj'
file = 'assets/torus.obj'
file = 'assets/suzanne.obj'
vertices, faces, normals = read_mesh(file)
glClearColor(0.2, 0.3, 0.3, 1.0) 
glEnable(GL_DEPTH_TEST)


class App:
    def on_create(self):
        self.translation = projective_transform(
            linear=np.eye(3),
            scale=1,
            translation=[0,0,0],
            perspective=[0,0,0],
        )

        self.perspective = projective_transform(
            linear=np.eye(3),
            scale=2.5,
            translation=[0,0,0],
            perspective=[0.1,0.1,1.5],
        )

        self.angle = 2
        self.axis = np.array([1,1,1],dtype=float)

        # It's faster using matrix, because it leverages the numpy library
        # But I left this here as a demonstrations that we can just multiply 
        # the quaternions and vector as usual, no problem i.e q v q*
        self.use_quaternion_matrix = True 

        self.light_direction = np.array([0.0, 2.0, 1.0], dtype=float)
        self.light_direction /= np.linalg.norm(self.light_direction)

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

    def on_update(self):
        self.angle+=0.1
        q = Quaternion.from_axis_angle(axis=self.axis, angle=self.angle)
        if self.use_quaternion_matrix:
            rotation = q.to_matrix()
            M = self.perspective @ self.translation @ rotation
        else :
            M = self.perspective @ self.translation
        

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
            color = (255,123,123)
            normal = normals[index]
            intensity = np.dot(normal, self.light_direction)
            color = np.abs(intensity) * np.array(color)/256.0
            glColor3f(*color)
            glVertex3f(*v0)
            glVertex3f(*v1)
            glVertex3f(*v2)
        glEnd()




def main():
    app = App()
    app.on_create()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                app.on_event(event)
                pygame.quit()
        app.on_update()
        pygame.display.flip()

if __name__ == '__main__':
    main()