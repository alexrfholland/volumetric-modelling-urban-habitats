
import json
import numpy as np
from scipy.interpolate import interp1d


class Colormap:
    def __init__(self, json_file_path=None):
        if json_file_path is None:
            json_file_path = 'data/colors/categorical-maps.json'  # default path
        self._colormaps = self._load_colormaps(json_file_path)

    @staticmethod
    def _load_colormaps(json_file_path):
        with open(json_file_path, 'r') as json_file:
            return json.load(json_file)

    def get_categorical_colors(self, n_colors, colormap_name=None):
        if colormap_name is None:  # if no specific colormap provided, use a default one
            colormap_name = list(self._colormaps.keys())[0]

        cm_data = np.array(self._colormaps[colormap_name])

        return cm_data[:n_colors].tolist()  # get the first n colors

    def get_continuous_colors(self, n_colors, colormap_name=None):
        if colormap_name is None:  # if no specific colormap provided, use a default one
            colormap_name = list(self._colormaps.keys())[0]

        cm_data = np.array(self._colormaps[colormap_name])

        color_func = interp1d(np.linspace(0, 1, cm_data.shape[0]), cm_data, axis=0)

        return color_func(np.linspace(0, 1, n_colors)).tolist()  # interpolate to get n colors

"""cm = Colormap()
import os
print(os.getcwd())
categorical_colors = cm.get_categorical_colors(5, 'batlow')
print(categorical_colors)

continuous_colors = cm.get_continuous_colors(5, 'batlow')
print(continuous_colors)"""


