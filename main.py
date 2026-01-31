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
        Calculates Vertex Normals.
        Handles Triangles (3 vertices), Quads (4 vertices), and Polygons.
        """
        # Initialize normals array
        normals = np.zeros((len(self.vertices), 3), dtype=np.float32)

        for face in self.faces:
            # 1. Get all vertices for the current face
            # We use float64 to prevent the previous 'division' error you saw
            verts = [np.array(self.vertices[idx - 1], dtype=np.float64) for idx in face]

            # We need at least 3 vertices to define a plane (and thus a normal)
            if len(verts) < 3:
                continue

            # 2. Calculate Face Normal
            # We only need the first 3 vertices to calculate the cross product
            v0, v1, v2 = verts[0], verts[1], verts[2]
            
            normal = np.cross(v1 - v0, v2 - v0)
            
            # 3. Normalize the vector
            norm_val = np.linalg.norm(normal)
            if norm_val == 0: 
                continue # Skip degenerate faces (zero area)

            normal /= norm_val

            # 4. Add this normal to ALL vertices belonging to this face
            for idx in face:
                normals[idx - 1] += normal

        # 5. Final Normalization of Vertex Normals (Average)
        # Calculate lengths of the accumulated vectors
        norms = np.linalg.norm(normals, axis=1)[:, np.newaxis]
        
        # Avoid division by zero for isolated vertices
        norms[norms == 0] = 1.0 
        
        normals /= norms
        self.normals = normals

    def write_obj(self, filename="Output.obj", normals=False):
        print(f"Writing to {filename}")

        if not filename.endswith(".obj"):
            filename = filename + ".obj"

        # Calculate normals if requested but missing
        if normals and len(self.normals) == 0:
            self.calculate_normals()
    
        with open(filename, 'w') as f:
            # 1. Write vertices
            for v in self.vertices:
                f.write(f'v {v[0]} {v[1]} {v[2]}\n')
            
            # 2. Write normals (if present and requested)
            has_normals = (len(self.normals) > 0) and normals
            if has_normals:
                for n in self.normals:
                    f.write(f'vn {n[0]} {n[1]} {n[2]}\n')

            # 3. Write Edges (Wireframe lines)
            if len(self.edges) != 0:
                for edge in self.edges:
                    # OBJ lines use 1-based indexing
                    f.write(f"l {edge[0] + 1} {edge[1] + 1}\n")

            # 4. Write faces (Dynamic Handling)
            for face in self.faces:
                # We build the string list for the current face
                face_strings = []
                
                for idx in face:
                    if has_normals:
                        # Format: vertex_index//normal_index
                        # (In this simple implementation, vertex_idx == normal_idx)
                        face_strings.append(f"{idx}//{idx}")
                    else:
                        # Format: vertex_index
                        face_strings.append(f"{idx}")
                
                # Join them with spaces and write
                f.write(f'f {" ".join(face_strings)}\n')
    
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

    def __add__(self, other):
        """
        Merges this plot with another plot (Union).
        Usage: combined_plot = plot1 + plot2
        """
        if not isinstance(other, WorkShape):
            raise TypeError("You can only add two s3dplot/WorkShape objects together.")

        # Create a new instance to hold the result
        result = s3dplot()
        
        # Copy Self Data (Base)
        result.vertices = list(self.vertices)
        result.faces = list(self.faces)
        result.edges = list(self.edges)
        result.normals = list(self.normals) # If they exist

        # Vertices count is needed to shift indices
        vert_offset = len(self.vertices)

        # -- Vertices: Just append
        result.vertices.extend(other.vertices)

        for face in other.faces:
            shifted_face = [idx + vert_offset for idx in face]
            result.faces.append(shifted_face)

        for edge in other.edges:
            shifted_edge = (edge[0] + vert_offset, edge[1] + vert_offset)
            result.edges.append(shifted_edge)
            
        return result
    
    def translate(self, x=0, y=0, z=0):
        """
        Moves ALL current vertices in the object by the given offset.
        Useful for moving a shape after you've already defined it.
        """
        new_verts = []
        for v in self.vertices:
            new_verts.append([v[0] + x, v[1] + y, v[2] + z])
        self.vertices = new_verts

    def plot(self, x, y, z, offset=(0,0,0)):
        """
        Draws a line with a global offset.
        """
        start_idx = len(self.vertices) 
        dx, dy, dz = offset
        
        for ix, iy, iz in zip(x, y, z):
            self.vertices.append([ix + dx, iy + dy, iz + dz])
            
        num_points = len(x)
        for i in range(num_points - 1):
            idx1 = start_idx + i
            idx2 = start_idx + i + 1
            self.edges.append((idx1, idx2))
    
    def sphere(self, cx=0, cy=0, cz=0, radius=1.0, rings=12, segments=12):
        """
        Generates a UV Sphere.
        rings: Number of horizontal slices (latitude)
        segments: Number of vertical slices (longitude)
        """
        base_idx = len(self.vertices) + 1  # OBJ is 1-based
        
        # 1. Generate Vertices
        for i in range(rings + 1):
            phi = np.pi * i / rings
            for j in range(segments + 1):
                theta = 2 * np.pi * j / segments
                
                x = cx + radius * np.sin(phi) * np.cos(theta)
                y = cy + radius * np.sin(phi) * np.sin(theta)
                z = cz + radius * np.cos(phi)
                self.vertices.append([x, y, z])

        # 2. Generate Faces (Quads)
        # We loop through the rings and segments to connect the vertices
        for i in range(rings):
            for j in range(segments):
                # Calculate indices of the 4 corners of the quad
                # Current Row
                p1 = base_idx + (i * (segments + 1)) + j
                p2 = p1 + 1
                # Next Row
                p3 = base_idx + ((i + 1) * (segments + 1)) + j
                p4 = p3 + 1
                
                # Add face (Counter-Clockwise winding order)
                self.faces.append([p1, p2, p4, p3])

    def add_cylinder(self, x=0, y=0, z=0, radius=0.5, height=1.0, segments=12):
        """
        Creates a cylinder with origin at the bottom center.
        """
        base_idx = len(self.vertices) + 1
        
        # 1. Generate Bottom Circle Vertices
        for i in range(segments):
            theta = 2.0 * np.pi * i / segments
            vx = x + radius * np.cos(theta)
            vy = y + radius * np.sin(theta)
            self.vertices.append([vx, vy, z]) # Bottom ring

        # 2. Generate Top Circle Vertices
        for i in range(segments):
            theta = 2.0 * np.pi * i / segments
            vx = x + radius * np.cos(theta)
            vy = y + radius * np.sin(theta)
            self.vertices.append([vx, vy, z + height]) # Top ring

        # 3. Side Faces
        for i in range(segments):
            # Bottom vertices are 0 to segments-1
            # Top vertices are segments to 2*segments-1
            
            b1 = base_idx + i
            b2 = base_idx + (i + 1) % segments # Wrap around to 0
            
            t1 = b1 + segments
            t2 = b2 + segments
            
            self.faces.append([b1, b2, t2, t1])

        # 4. Cap Faces (Simple N-gon fan or center point)
        # For simplicity, we are leaving caps open here, but you can 
        # add a center vertex and fan triangles if needed.
        
    def cuboid(self,x=0,y=0,z=0, size=1.0) -> None: # To Do: Make size a Range
        # Is Basically Voxel Mesh
        """_summary_
  
        Makes a 3D cubiod on the given coordinate as the origin, the size of cuboid is 1 unit.
        
        Args:
            x,y,z (_type_): int
            size (_type_): float
        Returns:
            Vertices and Faces
        """
        #for x, y, z, size in zip(X, Y, Z, size):  To Do: Make size variable for repeating Values, would probably have to make an HashFunction for x,y,z value

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

    def scatter(self, x, y, z, s=1.0, marker='cube', *args, offset=(0,0,0)):
        """
        offset: tuple (dx, dy, dz) to shift the entire plot.
        """
        if isinstance(x, (int, float)): x, y, z = [x], [y], [z]
        
        dx, dy, dz = offset
        marker_offset = s / 2.0
        
        for ix, iy, iz in zip(x, y, z):
            final_x = ix + dx
            final_y = iy + dy
            final_z = iz + dz
            
            if marker == 'cube':
                self.cuboid(final_x - marker_offset, 
                            final_y - marker_offset, 
                            final_z - marker_offset, size=s)
            elif marker == 'sphere':
                self.sphere(final_x, final_y, final_z, radius=s/2)

    def voxel_mesh(self,X=0,Y=0,Z=0, size=[1.0]):
        size = size[0]
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
    
    def bar_mesh(self, values, width=1, depth=1, space=0):
        """
        Add Bar 3D plot.
        Args:
            values (list): The height values of the categories.
            width (float or list): Width of bars.
            depth (float or list): Depth (thickness) of bars.
            space (float or list): Gap between bars.
        """
        n = len(values)
        index = 0
        
        # FIX: Check for both int AND float
        if isinstance(width, (int, float)):
            width = [width] * n
        if isinstance(depth, (int, float)):
            depth = [depth] * n
        if isinstance(space, (int, float)):
            space = [space] * n

        current_x = 0
        
        for i in range(n):
            # Calculate position based on previous placement
            # Using current_x ensures bars don't overlap if widths vary
            self.add_bar(index, current_x, 0, width[i], values[i], depth[i])
            
            # Increment X for the next bar
            # (current width + spacing)
            current_x += width[i] + space[i]
            
            # Each bar adds 8 vertices, so we shift the base index
            index += 8

    def clear(self):
        self.faces = []
        self.vertices = []
        self.normals = []
        self.edges = []

    def regular_face_plot(self, x=None, y=None, z=None, offset=(0,0,0)): 
        """
        offset: tuple (dx, dy, dz) to shift the whole surface.
        """
        n_rows, n_cols = x.shape
        start_v = len(self.vertices) 
        dx, dy, dz = offset

        # Add Vertices with Offset
        for i in range(n_rows):
            for j in range(n_cols):
                self.vertices.append([
                    x[i, j] + dx, 
                    y[i, j] + dy, 
                    z[i, j] + dz
                ])
  
        # Add Faces (Indices logic remains the same)
        for i in range(n_rows - 1):
            for j in range(n_cols - 1):
                v0 = start_v + i * n_cols + j + 1
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

    def fill_between(self, x1, y1, z1, *args,x2=None, y2=None, z2=None, offset=(0,0,0)):
        """
        Creates a surface connecting two arbitrary 3D curves.
        
        Args:
            x1, y1, z1: Coordinates of the first curve.
            x2, y2, z2: Coordinates of the second curve. 
                        - If x2 is None, it defaults to x1 (vertical fill).
                        - If y2 is None, it defaults to y1.
                        - If z2 is None, it defaults to 0 (floor fill).
            offset: (dx, dy, dz) Global shift for the entire shape.
        """
        n = len(x1)
        dx, dy, dz = offset

        # --- Handle Defaults for Curve 2 ---
        
        # If x2 is missing, assume it shares x1 (vertical/depth alignment)
        if x2 is None: x2 = x1
        
        # If y2 is missing, assume it shares y1
        if y2 is None: y2 = y1
        
        # If z2 is missing, assume it fills down to 0
        if z2 is None: 
            z2 = [0] * n
        elif isinstance(z2, (int, float)): 
            z2 = [z2] * n

        # Check for length mismatch
        if len(x2) != n:
            print(f"Warning: Curve 1 has {n} points but Curve 2 has {len(x2)}. Truncating to match.")
            n = min(n, len(x2))

        # --- 1. Add Vertices ---
        base_idx = len(self.vertices) + 1
        
        # Curve 1 vertices
        for i in range(n):
            self.vertices.append([x1[i]+dx, y1[i]+dy, z1[i]+dz])
            
        # Curve 2 vertices
        for i in range(n):
            self.vertices.append([x2[i]+dx, y2[i]+dy, z2[i]+dz])

        # --- 2. Create Faces ---
        # Connect i -> i (parallel/corresponding points)
        for i in range(n - 1):
            top_curr = base_idx + i
            top_next = base_idx + i + 1
            bot_curr = base_idx + n + i
            bot_next = base_idx + n + i + 1
            
            self.faces.append([top_curr, bot_curr, bot_next, top_next])
    
    def quiver(self, x, y, z, u, v, w, length=1.0, arrow_length_ratio=0.3, offset=(0,0,0)):
        if isinstance(x, (int, float)): 
            x,y,z = [x],[y],[z]
            u,v,w = [u],[v],[w]

        dx, dy, dz = offset

        for i in range(len(x)):
            # Apply offset to the starting position
            start = np.array([x[i] + dx, y[i] + dy, z[i] + dz])
            
            # Direction calculation remains the same
            direction = np.array([u[i], v[i], w[i]])
            norm = np.linalg.norm(direction)
            if norm == 0: continue
            
            direction = direction / norm 
            end = start + (direction * length)
            
            # Shaft
            self.vertices.append(start.tolist())
            self.vertices.append(end.tolist())
            self.edges.append((len(self.vertices)-2, len(self.vertices)-1))

            # Head
            head_size = length * arrow_length_ratio
            base_center = end - (direction * head_size)
            
            if abs(direction[2]) < 0.9: perp = np.cross(direction, [0,0,1])
            else: perp = np.cross(direction, [0,1,0])
            perp = perp / np.linalg.norm(perp)
            perp2 = np.cross(direction, perp)
            
            w_head = head_size * 0.4
            
            # Head Vertices
            v_tip = end
            v_b1 = base_center + perp * w_head
            v_b2 = base_center - perp * w_head
            v_b3 = base_center + perp2 * w_head
            v_b4 = base_center - perp2 * w_head
            
            b_idx = len(self.vertices) + 1
            self.vertices.extend([v_tip.tolist(), v_b1.tolist(), v_b2.tolist(), v_b3.tolist(), v_b4.tolist()])
            
            # Head Faces
            self.faces.extend([
                [b_idx, b_idx+1, b_idx+3], [b_idx, b_idx+3, b_idx+2],
                [b_idx, b_idx+2, b_idx+4], [b_idx, b_idx+4, b_idx+1],
                [b_idx+1, b_idx+4, b_idx+2, b_idx+3] 
            ])
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
