from Config.config_loader import ConfigLoader
from Config.update_config import update_paths
import os

project_dir = os.path.abspath(os.path.dirname(__file__))
update_paths(project_dir)
config_path = os.path.join(project_dir, 'config.json')
print("Config file updated successfully.")
default_config = ConfigLoader(config_path).load_config()
