import json
import os


class ConfigLoader:
    def __init__(self, config_dir):
        """
        loads in config if no file path is spcified it loads it the path stored in config path. Must run update config
        for it to be accurate
        :param file_path:
        """
        self.config_dir = config_dir
        self.project_dir = os.path.dirname(config_dir)
        self.config_path = os.path.join(config_dir, 'config.json')
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
        return print("Config not found")

    def load_label_maps(self):
        label_map_dir = os.path.join(self.project_dir, 'LabelMaps')
        # Initialize an empty dictionary to hold the label maps
        label_maps = {}

        # Loop through each file in the LabelMaps directory
        for filename in os.listdir(label_map_dir):
            if filename.endswith(".txt"):
                filepath = os.path.join(label_map_dir, filename)

                # Read the content of the txt file into a list
                with open(filepath, 'r') as f:
                    lines = f.readlines()

                # Remove newline characters and store in dictionary
                label_maps[filename[:-4]] = [line.strip() for line in lines]

        # Add label maps to config
        self.config['label_maps'] = label_maps

    def update_config(self):

        with open(self.config_path, 'w') as file:
            json.dump(self.config, file, indent=4)

    def write_label_map(self, config_dir, class_lists=[]):
            for classes in class_lists:
                label_map_path = os.path.join(config_dir, f'label_map_{classes}.txt')
                self.config['label_maps'][classes] = label_map_path
                labels = self.config['class_lists'][classes]
                with open(label_map_path, 'w') as file:
                    for label in labels:
                        file.write(str(label) + '\n')
                self.update_config()

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
            config['model_paths'][key] = os.path.join(model_dir, filename)

        # updating config path assuming that is named config.json
        config['config_path'] = os.path.join(model_dir, 'config.json')

        # Write the updated config back to the file
        with open(self.config_path, 'w') as file:
            json.dump(config, file, indent=4)
        print("Config file updated successfully.")


