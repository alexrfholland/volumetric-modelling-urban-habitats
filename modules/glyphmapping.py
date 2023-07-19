import pyvista as pv
import numpy as np

glyphs_to_plot = []

def add_glyphs_to_visualiser(positions, sizes, colors, solid=True, line_width=1.0, cmap='rainbow'):
    # Create glyphs
    cube = pv.Cube() if solid else pv.Cube().outline()
    points = pv.PolyData(positions)
    points["sizes"] = sizes
    points["colors"] = colors  # this will be used as scalars for the cmap
    glyphs = points.glyph(scale="sizes", factor=1.0, geom=cube)

    # Add to global list
    settings = {
        'glyphs': glyphs,
        'solid': solid,
        'line_width': line_width,
        'cmap': cmap
    }
    glyphs_to_plot.append(settings)

def plot_glyphs():
    p = pv.Plotter()
    p.add_light(pv.Light(position=(2, 2, 0), focal_point=(0, 0, 0), intensity=.7))
    light2 = pv.Light(light_type='cameralight', intensity=.5)
    light2.specular = 0.5  # Reduced specular reflection
    p.add_light(light2)


    for setting in glyphs_to_plot:
        if setting['solid']:
            p.add_mesh(setting['glyphs'], scalars="colors", cmap=setting['cmap'], opacity=1.0)
        else:
            p.add_mesh(setting['glyphs'], scalars="colors", cmap=setting['cmap'], opacity=1.0, style='wireframe', line_width=setting['line_width'])

    p.enable_eye_dome_lighting()
    p.show()

# Sample Usage:

def main():
    num_points = 10000
    positions = np.random.rand(num_points, 3) * 5000
    sizes = np.random.rand(num_points)
    colors = np.random.rand(num_points)

    add_glyphs_to_visualiser(positions, sizes, colors, solid=True)
    add_glyphs_to_visualiser(positions/2, sizes*0.5, colors, solid=False, line_width=5, cmap='cool')
    
    plot_glyphs()

if __name__ == '__main__':
    main()
