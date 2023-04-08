import numpy as np

class Quaternion:
    def __init__(self, w=1.0, x=0.0, y=0.0, z=0.0):
        # I chose the w first, as it's the real part
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    @classmethod
    def from_axis_angle(cls,axis, angle):
        """Function that creates a quaternion from an axis and a angle"""
        # We need to normalize the axis
        axis /= np.linalg.norm(axis)
        
        # Calculates cos and sin, just once. Using by HALF the angle given
        cos = np.cos(angle/2.0)
        sin = np.sin(angle/2.0)
        # Real part gets cos
        w = cos
        # The vector part get sin plus the axis
        x = sin * axis[0]
        y = sin * axis[1]
        z = sin * axis[2]
        # Create the quaternions and return
        q = cls(w,x,y,z)
        return q

    def multiply_quaternion(self, other):
        """Multiply two quaternions"""
        # Nothing too interesnting, just the expansion of terms and multiply
        w = self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z
        x = self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y
        y = self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x
        z = self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w
        return Quaternion(w, x, y, z)

    def rotate(self, vec):
        """ Rotate a vertor 'v' by a quaternion 'q' as usual q v q* """
        # Quaternion 'q' is the self object
        q = self
        # Interpret v as a vector only part Quaternion
        v = Quaternion(0.0, vec[0], vec[1], vec[2])
        # Do the multiply
        v = q * v * q.conjugate()

        # Modify the vector inplace, this is interesting 
        # when the vector might be in homogenous coordinates
        vec[0] = v.x
        vec[1] = v.y
        vec[2] = v.z
        return  vec

    def conjugate(self):
        """ Conjugates Quaternion normally """
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def is_unitary(self):
        norm = self.norm()
        unitary = np.isclose(norm, 1.0)
        return unitary
    
    def to_matrix(self):
        """If unitary is set to true, that means we can use the more concise matrix, and assume norm == 1"""
        w,x,y,z = self.w,self.x,self.y,self.z
        # So we always check is is unitary just for debuggin
        # because probably we don't wanna transform a quaternion
        # to a matrix when the quaternion isn't unitary
        if not self.is_unitary():
            raise Exception("Quaternion ain't unitary, we don't support matrix representation of abitrary quaternions (LAZY)") 
        # Just using the definition used in the Text Book
        # to convert to matrix, and also we make it homogenous matrix
        return np.array([
            [1 - 2 * (y**2 + z**2),  2 * (x*y - w*z),       2 * (x*z + w*y),       0],
            [2 * (x*y + w*z),        1 - 2 * (x**2 + z**2), 2 * (y*z - w*x),       0],
            [2 * (x*z - w*y),        2 * (y*z + w*x),       1 - 2 * (x**2 + y**2), 0],
            [0,                      0,                     0,                     1]
        ], dtype=float)

    def norm(self):
        return np.sqrt(self.w ** 2 + self.x ** 2 + self.y ** 2 + self.z ** 2)

    def normalize(self):
        norm = self.norm()
        if norm != 0:
            self.x /= norm
            self.y /= norm
            self.z /= norm
            self.w /= norm

    # Below theres Python Specific "dunder" metohds, to facilitate using this class
    # such as conversion to string str(q), equality check q1 == q2, multiply sintax q1 * q2
    # add, subtraction. and thats it

    def __str__(self):
        return f"Quaternion({self.x}, {self.y}, {self.z}, {self.w})"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z and self.w == other.w
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

