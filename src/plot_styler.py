import json
import matplotlib.pyplot as plt
from matplotlib import rcParams

class PlotStyler:
    def __init__(self, config_path:str='../data/plot_config.json'):
        """
        Initialize the PlotStyler with a configuration file.
        :param config_path: Path to the JSON file containing color and style settings.
        """
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Apply global matplotlib settings
        rcParams['axes.facecolor'] = self.config['background_color']
        rcParams['axes.edgecolor'] = self.config['grid_color']
        rcParams['grid.color'] = self.config['grid_color']
        rcParams['axes.titleweight'] = 'bold'
        rcParams['axes.labelweight'] = 'bold'
        rcParams['axes.titlesize'] = 14
        rcParams['axes.labelsize'] = 12
        rcParams['font.family'] = 'Calibri'
        rcParams['text.color'] = self.config['label_color']
        rcParams['axes.labelcolor'] = self.config['label_color']
        rcParams['xtick.color'] = self.config['label_color']
        rcParams['ytick.color'] = self.config['label_color']

    def apply_style(self, ax, title, xlabel, ylabel):
        """
        Apply consistent styling to a given plot.
        :param ax: Matplotlib Axes object.
        :param title: Title of the plot (converted to all caps).
        :param xlabel: Label for the x-axis (converted to all caps).
        :param ylabel: Label for the y-axis (converted to all caps).
        """
        ax.set_title(title.upper(), color=self.config['title_color'])
        ax.set_xlabel(xlabel.upper(), color=self.config['label_color'])
        ax.set_ylabel(ylabel.upper(), color=self.config['label_color'])
        ax.grid(True, linestyle='--', linewidth=0.5)

    def get_color(self, color_name):
        """
        Get a color from the configuration file.
        :param color_name: Name of the color (key in the JSON file).
        :return: Hex color code as a string.
        """
        return self.config.get(color_name, '#000000')  # Default to black if not found