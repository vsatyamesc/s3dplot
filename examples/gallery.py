import numpy as np
import s3dplot

# This will hold our master scene
gallery = s3dplot()

# --- 1. GEOMETRIC PRIMITIVES ---

# Standard Cube
box = s3dplot()
box.cuboid(size=2)
box.translate(x=0, y=0, z=0) 
gallery = gallery + box

# UV Sphere
ball = s3dplot()
ball.sphere(radius=1.5, rings=16, segments=16)
ball.translate(x=5, y=0, z=0)
gallery = gallery + ball

# Cylinder 
cyl = s3dplot()
cyl.add_cylinder(radius=0.8, height=3, segments=12)
cyl.translate(x=10, y=0, z=0)
gallery = gallery + cyl

# Voxel Mesh (Minecraft style)
vox = s3dplot()
vx = [0, 1, 0, 0, 0] # relative x positions
vy = [0, 0, 1, -1, 0] # relative y positions
vz = [0, 0, 0, 0, 1] # relative z positions
vox.voxel_mesh(vx, vy, vz, size=[0.8])
vox.translate(x=15, y=0, z=0)
gallery = gallery + vox


# --- 2. DATA VISUALIZATION ---

# Scatter Plot (Cube markers)
scat1 = s3dplot()
x = np.random.uniform(-1, 1, 15)
y = np.random.uniform(-1, 1, 15)
z = np.random.uniform(0, 3, 15)
scat1.scatter(x, y, z, s=0.3, marker='cube')
scat1.translate(x=0, y=10, z=0)
gallery = gallery + scat1

# Scatter Plot (Sphere markers)
scat2 = s3dplot()
scat2.scatter(x, y, z, s=0.4, marker='sphere')
scat2.translate(x=5, y=10, z=0)
gallery = gallery + scat2

# 3D Bar Chart
bars = s3dplot()
values = [1, 4, 2, 5, 3]
# width can be a float or a list of floats
bars.bar_mesh(values, width=0.6, depth=0.6, space=0.2)
bars.translate(x=10, y=10, z=0)
gallery = gallery + bars

# Line Plot (Trajectory string)
line = s3dplot()
t = np.linspace(0, 10, 50)
lx = np.sin(t)
ly = np.cos(t)
lz = t / 2
line.plot(lx, ly, lz)
line.translate(x=15, y=10, z=0)
gallery = gallery + line


# --- 3. SURFACES & TOPOLOGY ---

# Regular Surface (Solid skin)
surf = s3dplot()
X, Y = np.meshgrid(np.linspace(-2, 2, 10), np.linspace(-2, 2, 10))
Z = np.sin(np.sqrt(X**2 + Y**2))
surf.regular_face_plot(X, Y, Z)
surf.translate(x=0, y=20, z=0)
gallery = gallery + surf

# Wireframe Surface (Net / Grid)
wire = s3dplot()
wire.regular_wireframe_plot(X, Y, Z)
wire.translate(x=5, y=20, z=0)
gallery = gallery + wire

# Fill Between (Ribbon connecting two curves)
ribbon = s3dplot()
fx = np.linspace(0, 5, 20)
fy = np.sin(fx)
fz_top = np.ones_like(fx) * 2 
fz_bot = np.zeros_like(fx)
# We connect the top curve to the bottom curve
ribbon.fill_between(fx, fy, fz_top, y2=fy, z2=fz_bot)
ribbon.translate(x=10, y=20, z=0)
gallery = gallery + ribbon

# Twisted Ribbon (Connecting non-aligned curves)
twist = s3dplot()
tx = np.linspace(0, 5, 20)
# Curve 1: straight line at y=0
twist.fill_between(tx, np.zeros(20), np.zeros(20), 
                   x2=tx, y2=np.ones(20), z2=np.sin(tx)) # Curve 2: sine wave at y=1
twist.translate(x=15, y=20, z=0)
gallery = gallery + twist


# --- 4. PHYSICS & ADVANCED ---

# Quiver (Vector Field)
quiv = s3dplot()
qx, qy, qz = [0,0,0], [1,0,0], [0,1,0] # Origin points
qu, qv, qw = [0,0,1], [0,0,1], [0,0,1] # Upward vectors
quiv.quiver(qx, qy, qz, qu, qv, qw, length=1.5)
quiv.translate(x=0, y=30, z=0)
gallery = gallery + quiv

# Magnetic Field Simulation (Dipole)
mag = s3dplot()
# Add a magnet cylinder
mag.add_cylinder(radius=0.2, height=2, x=0, y=0, z=-1)
# Create a small grid around it
gx, gy, gz = np.meshgrid([-1, 1], [-1, 1], [-1, 0, 1])
flat_x, flat_y, flat_z = gx.flatten(), gy.flatten(), gz.flatten()
# Simple outward flow logic for demo
mag.quiver(flat_x, flat_y, flat_z, flat_x, flat_y, flat_z, length=0.5)
mag.translate(x=5, y=30, z=0)
gallery = gallery + mag

# --- EXPORT ---

print("Calculating normals for the entire gallery...")
gallery.calculate_normals()

filename = "FullGallery.obj"
print(f"Saving to {filename}...")
gallery.write_obj(filename, normals=True)
print("Done! Open the OBJ file to see your library in action.")