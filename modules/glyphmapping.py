import numpy as np
import pyvista as pv


glyphs_to_plot = []

# 'glyphs_to_plot' is a list that contains dictionaries. Each dictionary represents a set of glyphs that 
# will be added to the visualiser. The dictionary has the following structure:

# 'glyphs': a pv.PolyData or pv.UnstructuredGrid object that contains the glyph geometry and associated data. 
# This is created using the 'glyph' method of a pv.PolyData or pv.UnstructuredGrid object. The 'glyph' method 
# generates glyphs at the points specified in the 'points' parameter. The glyphs can be scaled and/or colored 
# according to scalar data associated with the points.

# 'solid': a boolean value that indicates whether the glyphs should be solid or not. If 'True', the glyphs 
# are solid. If 'False', only the outline of the glyphs is shown.

# 'line_width': a float value that specifies the line width for the glyphs when 'solid' is 'False'.

# 'cmap': a string that specifies the colormap to use for coloring the glyphs. This is used when the glyphs 
# are colored according to a scalar value. If 'rgb' is specified, the glyphs are colored according to RGB values.

# The structure of 'glyphs_to_plot' depends on which function is used to add the glyphs:

# 1) In 'add_glyphs_to_visualiser()', the 'glyphs' are created by scaling a cube geometry according to the 
# 'sizes' parameter and coloring the glyphs according to the 'color_scalar' parameter.

# 2) In 'add_voxels_with_rgba_to_visualiser()', the 'glyphs' are created by scaling a box geometry according 
# to the 'sizes' parameter and coloring the glyphs according to the 'colors' parameter.

# 3) In 'add_point_cloud_to_visualiser()', the 'glyphs' are points and are colored according to the 'colors' parameter.


# In the 'plot()' function, the glyphs are added to the plotter based on the settings in 'glyphs_to_plot':

# 1) If 'cmap' is 'rgb', then the glyphs are colored according to RGB values. The 'scalars' parameter of 
# 'add_mesh()' is set to "colors", which tells 'add_mesh()' to color the glyphs based on the RGB values in 
# the 'colors' array. If 'solid' is 'True', the glyphs are solid. If 'solid' is 'False', the glyphs are 
# wireframe and the line width is set according to 'line_width'.

# 2) If 'cmap' is not 'rgb' and 'sizes' is in the point data of the glyphs, then the glyphs are colored according
# to a scalar value. The 'scalars' parameter of 'add_mesh()' is set to "color_scalar", which tells 'add_mesh()' 
# to color the glyphs based on the scalar values in the 'color_scalar' array. The colormap is set according to 'cmap'. 
# If 'solid' is 'True', the glyphs are solid. If 'solid' is 'False', the glyphs are wireframe and the line width 
# is set according to 'line_width'.

# 3) If 'cmap' is not 'rgb' and 'sizes' is not in the point data of the glyphs, then the glyphs are added without 
# any scalar coloring.


def add_glyphs_to_visualiser(positions, sizes, color_scalar, solid=True, line_width=1.0, cmap='rainbow'):
    cube = pv.Cube() if solid else pv.Cube().outline()
    points = pv.PolyData(positions)
    points["sizes"] = sizes
    points["color_scalar"] = color_scalar
    glyphs = points.glyph(scale="sizes", factor=1.0, geom=cube)

    settings = {
        'glyphs': glyphs,
        'solid': solid,
        'line_width': line_width,
        'cmap': cmap
    }
    glyphs_to_plot.append(settings)

def add_voxels_with_rgba_to_visualiser(positions, sizes, colors, base_voxel_size=1.0):
    positions_np = np.array(positions)
    sizes_np = np.array(sizes)
    colors_np = np.array(colors)
    
    ugrid = pv.PolyData(positions_np)
    ugrid.point_data["sizes"] = sizes_np
    ugrid.point_data["colors"] = np.hstack((colors_np, np.ones((colors_np.shape[0], 1))))
    
    glyph = pv.Box().scale(base_voxel_size / 2.0, base_voxel_size / 2.0, base_voxel_size / 2.0)
    
    settings = {
        'glyphs': ugrid.glyph(geom=glyph, scale="sizes"),
        'solid': True,
        'line_width': 1.0,
        'cmap': 'rgb'
    }
    glyphs_to_plot.append(settings)


def add_point_cloud_to_visualiser(positions, sizes, colors):
    points = pv.PolyData(positions)
    points["sizes"] = sizes
    points["colors"] = colors
    
    settings = {
        'glyphs': points,
        'solid': True,
        'cmap': 'rgb'
    }
    glyphs_to_plot.append(settings)

def plot():
    p = pv.Plotter()
    p.add_light(pv.Light(position=(2, 2, 0), focal_point=(0, 0, 0), intensity=.7))
    light2 = pv.Light(light_type='cameralight', intensity=.5)
    light2.specular = 0.5  # Reduced specular reflection
    p.add_light(light2)

    for setting in glyphs_to_plot:

        # If colormap is set to 'rgb', glyphs are colored according to RGB values
        if setting['cmap'] == 'rgb':

            # If 'solid' is True, glyphs are solid and fully opaque 
            if setting['solid']:
                p.add_mesh(setting['glyphs'], scalars="colors", rgb=True, opacity=1.0)

            # If 'solid' is False, glyphs are wireframe with specified line width 
            else:
                p.add_mesh(setting['glyphs'], scalars="colors", rgb=True, opacity=1.0, style='wireframe', line_width=setting['line_width'])

        # If colormap is not set to 'rgb' and 'sizes' are included in the glyph's point data, glyphs are colored 
        # according to a scalar value
        elif 'sizes' in setting['glyphs'].point_data:

            # If 'solid' is True, glyphs are solid and semi-transparent
            if setting['solid']:
                p.add_mesh(setting['glyphs'], scalars="color_scalar", cmap=setting['cmap'], opacity=.07)

            # If 'solid' is False, glyphs are wireframe with specified line width 
            else:
                p.add_mesh(setting['glyphs'], scalars="color_scalar", cmap=setting['cmap'], opacity=1.0, style='wireframe', line_width=setting['line_width'])

        # If colormap is not set to 'rgb' and 'sizes' are not included in the glyph's point data, glyphs are added 
        # without any scalar coloring
        else:
            p.add_mesh(setting['glyphs'], opacity=1.0)

    p.enable_eye_dome_lighting()
    p.show()



def main():
    num_points = 10000
    
    positions = np.random.rand(num_points, 3) * 5
    sizes = np.random.rand(num_points)
    color_scalars = np.random.rand(num_points)  # Generating random scalar values for coloring
    colors = np.ones((num_points, 3)) * [1, 0, 0]  # red colors


    # Add glyphs using the add_glyphs_to_visualiser function
    add_glyphs_to_visualiser(positions, sizes, color_scalars, solid=False, line_width=20, cmap='viridis')
    #add_voxels_with_rgba_to_visualiser(positions, sizes, colors)
    #add_point_cloud_to_visualiser(positions, sizes, colors)
    
  

    plot()

if __name__ == '__main__':
    main()


