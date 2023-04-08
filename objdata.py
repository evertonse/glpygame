from dataclasses import dataclass 
from pathlib import Path
import numpy as np

@dataclass
class ObjData:
    vertices: list[list[float]] 
    faces:  list[list[int]] 

    def from_obj(filename:str):
        """Parse the .obj file format into vertices and faces"""
        vertices = []
        faces = []

        with open(file=filename) as objfile:
            for line in objfile.readlines():
                line = line.replace('\n','').strip()
                if line.startswith('#'):
                   continue 
                if line.startswith('vt'):
                   continue 
                if line.startswith('vn'):
                    continue
                elif line.startswith('v'):
                    vertex = [float(v) for v in line.split()[1:]]
                    assert(len(vertex) == 3 )
                    vertices.append(vertex)
                elif line.startswith('f'):
                    face = [int(f.split('/')[0]) - 1 for f in line.split()[1:]]
                    # Might have faces that are made with more than 3 vertices
                    # we shoud just convert into more faces of 3 , using the algorithm 
                    # I use in the .off parsing
                    assert(len(face) == 3 )
                    faces.append(face)
                else:
                    pass #ignore it for now,
        return ObjData(vertices=vertices, faces=faces) 
    
    def from_off(filename:str):
        """Parse the .off file format into vertices and faces"""
        try:
            with open(filename, 'r') as file:
                # First line tell us the file format 
                file_format = file.readline().strip()
                if file_format.lower().count("off") == 0:
                    raise ValueError("Invalid file format. Expected OFF, got " + file_format)

                # Next line tell us the number of vertices and faces
                n_vertices, n_faces, n_edges = file.readline().strip().split()

                # Vertices Parsing 
                vertices = []
                for _ in range(int(n_vertices)):
                    line = file.readline()
                    if not line:
                        raise ValueError("Unexpected end of file while reading vertices")
                    vertex = list(map(float, line.strip().split()))
                    if len(vertex) != 3:
                        raise ValueError(f"Invalid vertex dimension, expected 3 got {len(vertex)}, content: " + line)
                    vertices.append(vertex)

                # Faces Parsing 
                faces = []
                for _ in range(int(n_faces)):
                    line = file.readline()
                    if not line:
                        raise ValueError("Unexpected end of file while reading faces")
                    n_vert_per_face, *face = list(map(int, line.strip().split()))
                    if n_vert_per_face < 3:
                        raise ValueError("Invalid face format: " + line)
                    if n_vert_per_face > 3:
                        # Convert face with more than 3 vertices into triangles, using easiest way I know
                        for j in range(n_vert_per_face  - 2):
                            triangle = [face[0], face[j + 1], face[j + 2]]
                            faces.append(triangle)
                    else:
                        faces.append(face)
                return ObjData(vertices=vertices, faces=faces) 

        except FileNotFoundError:
            raise FileNotFoundError(f"FileNotFoundError {e}")
        except ValueError as e:
            raise ValueError(f"Value error {e}")
        
def read_mesh(filepath, homogenous=False):
    """Read a filepath and parse .obj or .off files, this function adds normals
      and can optionally convert vertices into homogenous coordinates 
    """
    ext = Path(filepath).suffix.lower()
    # Checking supported types
    if ext in {'.off', 'off'}:
        obj = ObjData.from_off(filepath)
    elif ext in {'.obj', 'obj'}:
        obj = ObjData.from_obj(filepath)
    else:
        raise Exception("We Only support .obj or .off files")
    
    vertices = np.array(obj.vertices,dtype=np.float32)
    faces = np.array(obj.faces,dtype=int)  

    normals = []
    for face in faces:
        # We manufacture normals gere
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        normal = np.cross(v2 - v0, v1 - v0)
        normal /= np.linalg.norm(normal)
        normals.append(normal)
    normals = np.array(normals,dtype=np.float32)  

    abs_max = np.abs(vertices.max())
    abs_min = np.abs(vertices.min())
    max_val = max(abs_max, abs_min)
    vertices = vertices/max_val

    if homogenous:
        # Add a bunch of ones if we want vectors in homogenous coordinates
        ones = np.ones(vertices.shape[0],dtype=np.float32)[:,np.newaxis]
        vertices = np.append(vertices, ones, axis=1)

    return vertices, faces, normals