# s3dplot (Only a Class not a module)
For Plotting, kind of more for creating 3D mesh for your matplotlib 3d plots. Use this just like your matplotlib code. Calculate whatever you want like you did in Numpy and pass the values. This is a newbie attempt. Has Docstrings to help in VS Code.
## Why :interrobang:
I once plotted a 3D graph and wanted to export it to use on a 3D software like Blender, but Matplotlib didn't give that option. So I've written a code that kind of works like Matplotlib.
## :closed_book: Important Note
This will only export the mesh, it will not add anything extra like legends, texts, colors (will add later), plot numbers, or any texts. You need to do that on your own in the current version.
## Example Usage :placard: Will add more examples.
```
from .main import s3dplot
import axes3d
X, Y, Z = axes3d.get_test_data(0.05) #get test data
plt = s3dplot()
plt.regular_wireframe_plot(X,Y,Z) #Plotting Wireframe plot
plt.write_obj("wireframe.obj")
```
## Plot Methods :robot:
:green_book: ```voxel_mesh``` to plot a voxel mesh. 
> plt.voxel_mesh(X=0,Y=0,Z=0, size=[1.0])

:green_book: ```bar_mesh``` to plot a bar mesh.
> plt.bar_mesh(values, width=1, depth=1, space=0)

```
Args:
  values (list or int): the values of the categories. (the height of the bar)
  width (list or int): the width you want to set, if different for each graph pass a list or tuple
  depth (list or int): the depth you want to set, if different for each graph pass a list or tuple
  space (list or int): spacing between bars, if different for each graph pass a list or tuple

If you didn't notice, since this is a 3d plot, you can see this 3d bar graph via different axes, thus each axis can have a certain value.
```

:green_book: ```regular_face_plot``` and ```regular_wireframe_plot``` Surface plot
> plt.regular_face_plot(x=[],y=[],z=[])

> plt.regular_wireframe_plot(x=[],y=[],z=[])
```
Takes an Array of X,Y,Z coordinates to plot surface usually X,Y is a meshgrid and Z is the Z coordinate

Args:
  x (_type_, optional): Mesh Grid
  y (_type_, optional): Mesh Grid
  z (_type_, optional):
```
