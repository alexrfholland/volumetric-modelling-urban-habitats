import numpy as np
import pyvista as pv


glyphs_to_plot = []

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
    print(glyphs_to_plot)


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
        
        print(setting['glyphs'].point_data)
        
        if setting['cmap'] == 'rgb':
            if setting['solid']:
                p.add_mesh(setting['glyphs'], scalars="colors", rgb=True, opacity=1.0)
            else:
                p.add_mesh(setting['glyphs'], scalars="colors", rgb=True, opacity=1.0, style='wireframe', line_width=setting['line_width'])
        
        elif 'sizes' in setting['glyphs'].point_data:
            if setting['solid']:
                p.add_mesh(setting['glyphs'], scalars="sizes", cmap=setting['cmap'], opacity=1.0)
            else:
                p.add_mesh(setting['glyphs'], scalars="sizes", cmap=setting['cmap'], opacity=1.0, style='wireframe', line_width=setting['line_width'])
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


