from Config.config_loader import ConfigLoader
import os

project_dir = os.path.abspath(os.path.dirname(__file__))
loader = ConfigLoader(project_dir)

loader.load_label_maps()
loader.update_paths()

default_config = loader.config
