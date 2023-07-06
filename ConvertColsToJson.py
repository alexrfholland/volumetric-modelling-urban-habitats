import os
import json
import importlib.util

def load_colormap_from_file(file_path):
    spec = importlib.util.spec_from_file_location("module.name", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.cm_data

def load_all_colormaps(root_dir):
    colormaps = {}
    for dir_name, sub_dir_list, file_list in os.walk(root_dir):
        for file_name in file_list:
            if file_name.endswith('.py') and not file_name.startswith('_'):
                file_path = os.path.join(dir_name, file_name)
                print(f"Loading colormap from: {file_path}")
                colormap_name = os.path.splitext(file_name)[0]  # get file name without extension
                colormap = load_colormap_from_file(file_path)
                print(f"Saving colormap as: {colormap_name}")  # print the name the colormap is saved as in the dictionary
                colormaps[colormap_name] = colormap  # store raw values

    return colormaps

def save_colormaps_to_json(colormaps, json_path):
    with open(json_path, 'w') as json_file:
        json.dump(colormaps, json_file)

colormaps = load_all_colormaps("colormaps")
save_colormaps_to_json(colormaps, "data/colors/categorical-maps.json")
