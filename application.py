# Import Pygame for Window management
import pygame
from pygame.locals import DOUBLEBUF,OPENGL 
# Import from pyonpengl to use opengl functions
from OpenGL.GL import *
from OpenGL.GLU import *

# Utility function for reading a mesh
from objdata import read_mesh 

# Numpy for you know what
import numpy as np
import numpy.linalg as la

# Projective Math has matrices for trnasformations
from projective_math import *
# My Implementatioon of Quaternions
from quaternion import Quaternion

# ALL options for rendering marked nas slow or fast rendering
# I recommend suzanne, because has an interesnting shape and
# it's fast. if you change the last value to another string
# it'll render another object
mesh_files = [
    'assets/rabbit.obj',    # Slow
    'assets/teapot.obj',    # Slow
    'assets/cat.obj',       # Slow 
    'assets/torus.obj',     # Fast
    'assets/cinda.obj',     # Fast
    'assets/samoyed.obj',   # Fast
    'assets/bunny.off',     # Slow 
    'assets/suzanne.obj',   # Fast
]
mesh_file = mesh_files[-1]

# Global variables for  Vertices Nomals and Faces, that were read 
# and parsed from mesh files 
vertices, faces, normals = read_mesh(mesh_file, homogenous=True)

class App:
    def on_create(self):
        # Translation, Perspective and Scale of Homogeneous Matrix
        self.translation = np.array([0.0, 0.0,  0.0], dtype=np.float32) 
        self.perspective = np.array([0.0, 0.05, 1.2], dtype=np.float32) 
        self.scale:float = 3.5

        # Axis and angle of quaternion rotation
        self.angle:float = 1.2
        self.axis = np.array([-1, -0.5, -1],dtype=np.float32)

        # Light rotation and direction to vabagundo shading
        self.light_direction = np.array([0.0, 2.0, 1.0], dtype=np.float32)
        self.light_rotation  = rotatex(0.15)

        # Color of the object being rendered
        self.mesh_color = (255,123,123)

        # Variable that controle if we should rotate 
        self.rotate = True 

        # It's faster using matrix, because it leverages the numpy library
        # But I left this here as a demonstrations that we can just multiply 
        # the quaternions and vector as usual, no problem i.e -> q v q*
        # if this variable is set to True it'll use the matrix, otherwise
        # it'll to they quaternion multiplication
        self.use_quaternion_matrix = True 


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

        clear_color = np.array([255,255,255,255])/255.0
        glClearColor(*clear_color) 
        glEnable(GL_DEPTH_TEST)

    def on_event(self,e) : 
        """Event Function, called on every event """

        # If pygame thing we should quit, we quit the whole app
        if e.type == pygame.QUIT:
            pygame.quit()
            exit()
        
        # For each key we control a boolean value
        # to decide if is being pressed right now, or not
        # this allows continues presing
        for key in self.is_down.keys():
            if e.type == pygame.KEYUP:
                if e.key == key:
                    self.is_down[key] = False
            if e.type == pygame.KEYDOWN:
                if e.key == key:
                    self.is_down[key] = True 

        if e.type == pygame.KEYDOWN:
            # If press escape it quits
            if e.key == pygame.K_ESCAPE:
                pygame.quit()
                exit()
            # If the key we pressed if Space,
            # then we stop the rotation
            if e.key == pygame.K_SPACE:
                self.rotate = not self.rotate

        # If we use the MouseWheel we change the scale of the
        # object being displayed
        if e.type == pygame.MOUSEWHEEL:
            if   e.y < 0:
                self.scale += 0.15 
            elif e.y > 0:
                self.scale -= 0.15

    def on_update(self):
        """ Update Function, called everyframe"""
        
        # Variable that contains information is a key is down
        is_down = self.is_down
        # We only continue rotation if rotate is set to True
        if self.rotate:
            self.angle += 0.025

        # Control Translation x-axis if 'w' or 's' is being pressed
        if is_down[pygame.K_w]:
            self.translation[1] += 0.1
        if is_down[pygame.K_s]:
            self.translation[1] -= 0.1

        # Control Translation y-axis if 'd' or 'a' is being pressed
        if is_down[pygame.K_d]:
            self.translation[0] += 0.1
        if is_down[pygame.K_a]:
            self.translation[0] -= 0.1

        # Control ligh direction with keys 'e' and 'q'
        if is_down[pygame.K_e]:
            self.light_direction = (np.array([*self.light_direction,1.0], dtype=np.float32) @ self.light_rotation)[:-1] 
        if is_down[pygame.K_q]:
            self.light_direction = (np.array([*self.light_direction,1.0], dtype=np.float32) @ self.light_rotation.T)[:-1] 

        # Control Perpective with 'x', 'z'
        if is_down[pygame.K_z]:
            self.perspective[2] += 0.035
        if is_down[pygame.K_x]:
            self.perspective[2] -= 0.035

        # Define our rotation quarternion given by axis and angle of rotation
        q = Quaternion.from_axis_angle(axis=self.axis, angle=self.angle)

        # Make a projective transformation wiht L = Identity
        # P as perspective, T as translation and S as scale
        # calculated from previously given values
        M = projective_transform(
            linear=np.eye(3),
            scale=self.scale,
            translation= self.translation,
            perspective= self.perspective
        )

        # If we're using the matrix representation oa a quarternion then we 
        # transform it right here and multiply with projective transformation 
        # Matrix M
        if self.use_quaternion_matrix:
            rotation = q.to_matrix()
            M = M @ rotation
 
        # Clear z-buffer and color-buffer
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        # Begin drawing Triangles
        glBegin(GL_TRIANGLES)
        for index,face in enumerate(faces):
            # Based on the faces we find the 0-th,1-th,2-th vertice
            # that makes up this triangle
            v0, v1, v2 = vertices[face]

            # If we opt for not using the matrix representation of quaternion
            # then we do the rotation by quaternion directly
            if not self.use_quaternion_matrix:
                v0 = q.rotate(v0)
                v1 = q.rotate(v1)
                v2 = q.rotate(v2)

            # Make the multiplication of matrix M for each vertice 
            v0 =  (M @ v0[:,np.newaxis]).flatten()
            v1 =  (M @ v1[:,np.newaxis]).flatten()
            v2 =  (M @ v2[:,np.newaxis]).flatten()

            # Perspective Divide
            v0 /= v0[-1]
            v1 /= v1[-1]
            v2 /= v2[-1]

            # Take only the first 3 coordinates
            v0 = v0[:-1]
            v1 = v1[:-1]
            v2 = v2[:-1]

            # Normalize light ray
            light_direction = self.light_direction/la.norm(self.light_direction)
            # Get the normal associated with this triangle face
            normal = normals[index]
            # Simple vabagundo shading, calculating intesity based on 
            # "sameness" of direction between light ray and normal
            intensity = np.dot(normal, light_direction)
            # Calculate the color with the new acquired intensity
            color = np.abs(intensity) * np.array(self.mesh_color)/256.0
            # Plot the triangle given by v0-v1-v2 and its color
            glColor3f(*color)
            glVertex3f(*v0)
            glVertex3f(*v1)
            glVertex3f(*v2)
        # End Plotting with open gl
        glEnd()


def main():
    # Init pygame to work with opengl
    pygame.init()
    # might need to change the window_size in case monitor isnt 600x600
    window_size = (600, 600)
    pygame.display.set_mode(window_size, DOUBLEBUF | OPENGL)

    # Create App and initialize
    app = App()
    app.on_create()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            # Call app's events
            app.on_event(event)
        # Update app
        app.on_update()
        # Flip the WHOLE buffer
        pygame.display.flip()

if __name__ == '__main__':
    main()