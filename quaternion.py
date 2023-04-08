import math
import numpy as np

class Quaternion:
    def __init__(self, w=1.0, x=0.0, y=0.0, z=0.0):
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    @classmethod 
    def from_axis_angle(cls,axis, angle):
        axis /= np.linalg.norm(axis)
        
        cos = np.cos(angle/2.0)
        sin = np.sin(angle/2.0)
        w = cos
        x = sin * axis[0]
        y = sin * axis[1]
        z = sin * axis[2]
        q = cls(w,x,y,z)
        return q

    def multiply_quaternion(self, other):
        w = self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z
        x = self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y
        y = self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x
        z = self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w
        return Quaternion(w, x, y, z)

    def rotate(self, vec):
        q = self
        v = Quaternion(0.0, vec[0], vec[1], vec[2])
        v = q * v * q.conjugate()
        vec[0] = v.x
        vec[1] = v.y
        vec[2] = v.z
        return  vec

    def conjugate(self):
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def is_unitary(self):
        norm = self.norm()
        unitary = np.isclose(norm, 1.0)
        return unitary
    
    def to_matrix(self):
        """If unitary is set to true, that means we can use the more concise matrix, and assume norm == 1"""
        w,x,y,z = self.w,self.x,self.y,self.z
        if not self.is_unitary():
            raise Exception("Quaternion ain't unitary, we don't support matrix representation of abitrary quaternions (LAZY)") 
        return np.array([
            [1 - 2 * (y**2 + z**2),  2 * (x*y - w*z),       2 * (x*z + w*y),       0],
            [2 * (x*y + w*z),        1 - 2 * (x**2 + z**2), 2 * (y*z - w*x),       0],
            [2 * (x*z - w*y),        2 * (y*z + w*x),       1 - 2 * (x**2 + y**2), 0],
            [0,                      0,                     0,                     1]
        ], dtype=float)

    def norm(self):
        return np.sqrt(self.w ** 2 + self.x ** 2 + self.y ** 2 + self.z ** 2)

    def __str__(self):
        return f"Quaternion({self.x}, {self.y}, {self.z}, {self.w})"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z and self.w == other.w

    def normalize(self):
        norm = self.norm()
        if norm != 0:
            self.x /= norm
            self.y /= norm
            self.z /= norm
            self.w /= norm

    def __mul__(self, other):
        if isinstance(other, Quaternion):
            return self.multiply_quaternion(other)
        elif isinstance(other, (int, float)):
            return Quaternion(self.w * other, self.x * other, self.y * other, self.z * other)
        raise Exception("danm don't know what the type is, panic mode !")


    def __rmul__(self, other):
       return Quaternion.__mul__(other,self) 

    def __add__(self, other):
        if isinstance(other, Quaternion):
            return Quaternion(self.w + other.w,self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        if isinstance(other, Quaternion):
            return Quaternion(self.w - other.w, self.x - other.x, self.y - other.y, self.z - other.z)

