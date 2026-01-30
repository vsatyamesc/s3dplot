import numpy as np

class PlotUtil():

    normals = []
    edges = []

    def __init__(self,face=None,vertices=None,edges=None,normals=None):
        self.faces = face or []
        self.vertices = vertices or []
        self.normals = normals or []
        self.edges = edges or []
  
    @staticmethod
    def help(self):
        print(f"Usually you shouldn't be here but you are that means you are looking for something.\nface =  A list of tuples/list, where each Element contains indices (1-based Index) of vertices that form each face.  \nvertices =  A list of tuples (x, y, z) representing the mesh vertices coordinates.\nnormals = A list of tuples (nx, ny, nz) representing the vertex normals, where n is a unit vector.")
        
    # face =  A list of tuples/list, where each Element contains indices (1-based Index) of vertices that form each face.
    # vertices =  A list of tuples (x, y, z) representing the mesh vertices coordinates.
    # normals = A list of tuples (nx, ny, nz) representing the vertex normals, where n is a unit vector.
  
    def calculate_normals(self) -> None:
        """
        Set Normals
        """
        normals = np.zeros((len(self.vertices), 3), dtype=np.float32)
        for face in self.faces:
            v0, v1, v2, _ = [np.array(self.vertices[idx - 1]) for idx in face]
            normal = np.cross(v1 - v0, v2 - v0)
            normal /= np.linalg.norm(normal)
            for idx in face:
                normals[idx - 1] += normal
        normals /= np.linalg.norm(normals, axis=1)[:, np.newaxis]
        self.normals = normals

    # @staticmethod
    # def calculate_normals(vertices,faces):
    #     normals = np.zeros((len(vertices), 3), dtype=np.float32)
    #     for face in faces:
    #         v0, v1, v2, _ = [np.array(vertices[idx - 1]) for idx in face]
    #         normal = np.cross(v1 - v0, v2 - v0)
    #         normal /= np.linalg.norm(normal)
    #         for idx in face:
    #             normals[idx - 1] += normal
    #     normals /= np.linalg.norm(normals, axis=1)[:, np.newaxis]
    #     return normals

    def write_obj(self, filename = "Output.obj", normals=False):

        print(f"Writing to {filename}")

        if not filename.endswith(".obj"):
            filename = filename + ".obj"

        if (normals == True) and len(self.normals)==0: # iff Normals are not provided during the object creation then calculate the normals if specified.
           self.calculate_normals()
    
        with open(filename, 'w') as f:
            # Write vertices
            for v in self.vertices:
                f.write(f'v {v[0]} {v[1]} {v[2]}\n')
            
            # Write normals if True and Present
            if len(self.normals) != 0 and (normals==True):
                for n in self.normals:
                    f.write(f'vn {n[0]} {n[1]} {n[2]}\n')

            # Write Edges if present, usually for Wireframes mode else, the face handles the edges
            if len(self.edges) != 0:
                for edge in self.edges:
                    f.write(f"l {edge[0] + 1} {edge[1] + 1}\n")

            # Write faces
            for face in self.faces:
                if (len(self.normals) != 0) and (normals==True):
                    f.write(f'f {face[0]}//{face[0]} {face[1]}//{face[1]} {face[2]}//{face[2]} {face[3]}//{face[3]}\n')
                else:
                    f.write(f'f {face[0]} {face[1]} {face[2]} {face[3]}\n')
    
    def write_stl(self, filename = "Output.stl"):
        # Occupied for Exporting in STL format
        pass
    
    def color_it(self):
        # Occupied for Vertex Painting the OBJ
        pass
  
class WorkShape():
    """_summary_
    Use New Instance of the Object Everytime you want to use a different Shape or it'll overwrite the previous data.
  
    Usually used to generate Vertices and Faces for the given Data
  
    Mainly for working with shapes, calculates vertices, faces based on predefined shapes and given coordinates. 
    You'll understand once you see the function.
    """
    def __init__(self):
        self.vertices = []
        self.faces = []
        self.edges = []
  
    def cuboid(self,x=0,y=0,z=0, size=1.0) -> None: # To Do: Make Size a Range
        # Is Basically Voxel Mesh
        """_summary_
  
        Makes a 3D cubiod on the given coordinate as the origin, the size of cuboid is 1 unit.
        
        Args:
            x,y,z (_type_): int
            size (_type_): float
        Returns:
            Vertices and Faces
        """
        #for x, y, z, size in zip(X, Y, Z, Size):  To Do: Make size variable for repeating Values, would probably have to make an HashFunction for x,y,z value

        base_index = len(self.vertices) + 1
        # Define vertices
        v0 = [x, y, z]
        v1 = [x + size, y, z]
        v2 = [x, y + size, z]
        v3 = [x + size, y + size, z]
        v4 = [x, y, z + size]
        v5 = [x + size, y, z + size]
        v6 = [x, y + size, z + size]
        v7 = [x + size, y + size, z + size]
        self.vertices.extend([v0, v1, v2, v3, v4, v5, v6, v7])
        # Add faces
        self.faces.extend([
                [base_index + 0, base_index + 1, base_index + 3, base_index + 2],  #Front
                [base_index + 4, base_index + 5, base_index + 7, base_index + 6],  #Back
                [base_index + 0, base_index + 1, base_index + 5, base_index + 4],  #Bottom
                [base_index + 2, base_index + 3, base_index + 7, base_index + 6],  # Top
                [base_index + 0, base_index + 2, base_index + 6, base_index + 4],  #Left
                [base_index + 1, base_index + 3, base_index + 7, base_index + 5]   #Right
        ])

    def voxel_mesh(self,X=0,Y=0,Z=0, Size=[1.0]):
        size = Size[0]
        for x, y, z in zip(X, Y, Z):
            self.cuboid(x, y, z, size)

    def add_bar(self, base_index=0,x=0,y=0, width=1, height=1, depth=1): # type: ignore

        vertices = self.vertices
        faces = self.faces

        v0 = [x, y, 0]
        v1 = [x + width, y, 0]
        v2 = [x, y + depth, 0]
        v3 = [x + width, y + depth, 0]
        v4 = [x, y, height]
        v5 = [x + width, y, height]
        v6 = [x, y + depth, height]
        v7 = [x + width, y + depth, height]

        vertices.extend([v0, v1, v2, v3, v4, v5, v6, v7])
        faces.extend([
            [base_index + 1, base_index + 2, base_index + 4, base_index + 3],  # Front
            [base_index + 5, base_index + 6, base_index + 8, base_index + 7],  # Back
            [base_index + 1, base_index + 2, base_index + 6, base_index + 5],  # Bottom
            [base_index + 3, base_index + 4, base_index + 8, base_index + 7],  # Top
            [base_index + 1, base_index + 3, base_index + 7, base_index + 5],  # Left
            [base_index + 2, base_index + 4, base_index + 8, base_index + 6]   # Right
        ])
    
    def bar_mesh(self,values, width=1, depth=1, space=0):
        """_summary_
        Add Bar 3D plot. This is a 3D Plot which means each plot can represent two different relations rather than one per graph unlike the 2D.
        Args:
            values (list or int): the values of the categories
            width (list or int): the width you want to set, if different for each graph pass a list or tuple
            depth (list or int): the depth you want to set, if different for each graph pass a list of tuple
            space (list or int): spacing, if different for each graph pass a list of tuple
        """
        index = 0
        n = len(values)
        if type(width) is int:
            width = [width]*n
        if type(depth) is int:
            depth = [depth]*n
        if type(space) is int:
            space = [space+1]*n
        for i in range(n):
            x = i*(width[i]*space[i])
            y = 0
            self.add_bar(index, x,y,width[i],values[i],depth[i])
            index += 8

    def clear(self):
        self.faces = []
        self.vertices = []
        self.normals = []
        self.edges = []

    def regular_face_plot(self,x=None,y=None,z=None): # Surface plot
        """_summary_  
        Takes an Array of X,Y,Z coordinates to plot surface
        usually X,Y is a meshgrid and Z is the Z coordinate
        Args: 
            x (_type_, optional): Mesh Grid
            y (_type_, optional): Mesh Grid
            z (_type_, optional): 
        """ 
        n_rows, n_cols = x.shape

        for i in range(n_rows):
          for j in range(n_cols):
            self.vertices.append([x[i, j], y[i, j], z[i, j]])
  
        for i in range(n_rows - 1):
          for j in range(n_cols - 1):
            v0 = i * n_cols + j + 1
            v1 = v0 + 1
            v2 = v0 + n_cols
            v3 = v2 + 1
            self.faces.append([v0, v1, v3, v2])
            
    def surface_plot(self,x=None,y=None,z=None):
        print("You could've literally used regular_face_plot bruh")
        self.regular_face_plot(x,y,z)

    def regular_wireframe_plot(self,x=None,y=None,z=None ): # To Do: Make it so that if any of the Axis is not provided then make it into an array of 0
        """_summary_
        Builds Edges and Vertices arrays with the given coordinates

        Args:
            x (_type_): 2d Array. Very important, if this is None, the Plane will be shifted
            y (_type_, optional): 2d Array. Defaults to None.
            z (_type_, optional): 2d Array. Defaults to None.
        """
        # X should be strictly an Array and not None
        # To Do: if X is none, change the planes

        n_rows, n_cols = x.shape
        if x is None and y is None and z is None:
            raise Exception("You didn't provide any of the Coordinates, Although this also means that faces and edges are empty.")

        # get the values for latter reason
        if x is not None:
            n_rows, n_cols = x.shape
        elif y is not None:
            n_rows, n_cols = y.shape
        elif z is not None:
            n_rows, n_cols = z.shape

        # Ensure x, y, and z are arrays
        if x is None:
            x = np.zeros((n_rows, n_cols))
        if y is None:
            y = np.zeros((n_rows, n_cols))
        if z is None:
            z = np.zeros((n_rows, n_cols))

        for i in range(n_rows):
          for j in range(n_cols):
            self.vertices.append([x[i, j], y[i, j], z[i, j]])

        # Add edges horizontally
        for i in range(n_rows):
            for j in range(n_cols - 1):
                index1 = i * n_cols + j
                index2 = index1 + 1
                self.edges.append((index1, index2))

        # Add edges vertically
        for i in range(n_rows - 1):
            for j in range(n_cols):
                index1 = i * n_cols + j
                index2 = index1 + n_cols
                self.edges.append((index1, index2))

    @staticmethod
    def get_midpoints(x): # should be a static method to get the center bc .
        """_summary_
        When working with voxel grids, you typically have the grid points located at the corners of the voxels, but for visualization and certain calculations (like rendering), you might want to refer to the center of each voxel. So, you need to calculate the midpoints before passing it to voxel_mesh
        Args:
            x (_type_): _description_
        Returns:
            _type_: _description_
        """
        sl = ()
        for _ in range(x.ndim):
            x = (x[sl + np.index_exp[:-1]] + x[sl + np.index_exp[1:]]) / 2.0
            sl += np.index_exp[:]
        return x
    
    def get_face(self) -> list:
        return self.faces
    
    def get_vert(self) -> list:
        return self.vertices
    
    def get_obj_data(self):
        return self.vertices, self.faces
      
class s3dplot(WorkShape,PlotUtil):
   pass
