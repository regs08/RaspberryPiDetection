from Config.config_loader import ConfigLoader
import os

project_dir = os.path.abspath(os.path.dirname(__file__))
loader = ConfigLoader(project_dir)

loader.update_paths()
loader.write_label_map(project_dir=project_dir, class_lists=['COCO'])

default_config = loader.config
