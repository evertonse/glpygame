from dataclasses import dataclass 

@dataclass
class ObjData:
    vertices: list[list[float]] 
    faces:  list[list[int]] 

    def from_obj(filename:str):
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
                    assert(len(face) == 3 )
                    faces.append(face)
                else:
                    pass #ignore it for now,
        return ObjData(vertices=vertices, faces=faces) 
    
    def from_off(filename:str):
        try:
            with open(filename, 'r') as file:
                # Read the first line to get the file format and the nber of vertices and faces
                file_format = file.readline().strip()
                n_vertices, n_faces, n_edges = file.readline().strip().split()
                if file_format.lower().count("off") == 0:
                    raise ValueError("Invalid file format. Expected OFF, got " + file_format)

                # Read the vertices
                vertices = []
                for _ in range(int(n_vertices)):
                    line = file.readline()
                    if not line:
                        raise ValueError("Unexpected end of file while reading vertices")
                    vertex = list(map(float, line.strip().split()))
                    if len(vertex) != 3:
                        raise ValueError(f"Invalid vertex dimension, expected 3 got {len(vertex)}, content: " + line)
                    vertices.append(vertex)

                # Read the faces
                faces = []
                for _ in range(int(n_faces)):
                    line = file.readline()
                    if not line:
                        raise ValueError("Unexpected end of file while reading faces")
                    n_vert_per_face, *face = list(map(int, line.strip().split()))
                    if n_vert_per_face < 3:
                        raise ValueError("Invalid face format: " + line)
                    if n_vert_per_face > 3:
                        # Convert face with more than 3 vertices into triangles
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
        