import json
import os


class ConfigLoader:
    def __init__(self, project_dir):
        """
        loads in config if no file path is spcified it loads it the path stored in config path. Must run update config
        for it to be accurate
        :param file_path:
        """
        self.project_dir = project_dir
        self.config_path = os.path.join(project_dir, 'config.json')
        self.config = self.load_config()

    def load_config(self):
        try:
            with open(self.config_path) as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"No config file found at {self.config_path}. Using default config.")
            return self.default_config()
        except json.JSONDecodeError:
            print(f"Failed to parse JSON from config file at {self.config_path}. Using default config.")
            return self.default_config()

    def default_config(self):
        return {
            'model_paths': {'yolov8n': '/default/path/here'},
            # Add your other default configurations here...
        }

    def write_label_map(self, project_dir, class_lists=[]):
            for classes in class_lists:
                label_map_path = os.path.join(project_dir, f'label_map_{classes}.txt')
                self.config['label_maps'][classes] = label_map_path
                labels = self.config['class_lists'][classes]
                with open(label_map_path, 'w') as file:
                    for label in labels:
                        file.write(str(label) + '\n')
                self.update_config()

    def update_config(self):

        with open(self.config_path, 'w') as file:
            json.dump(self.config, file, indent=4)

    def update_paths(self):

        # get project dir
        # where models are stored
        model_dir = os.path.join(self.project_dir, 'Models')

        # Read the existing config
        with open(self.config_path, 'r') as file:
            config = json.load(file)

        # Replace the paths
        for key, value in config['model_paths'].items():
            filename = os.path.basename(value)
            #new_path = value.replace("...", repo_path)
            config['model_paths'][key] = os.path.join(model_dir, filename)
        # updating config path assuming that is named config.json
        config['config_path'] = os.path.join(model_dir, 'config.json')
        # Write the updated config back to the file
        with open(self.config_path, 'w') as file:
            json.dump(config, file, indent=4)
        print("Config file updated successfully.")

