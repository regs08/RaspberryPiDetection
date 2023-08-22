import json


class ConfigLoader:
    def __init__(self, file_path):
        """
        loads in config if no file path is spcified it loads it the path stored in config path. Must run update config
        for it to be accurate
        :param file_path:
        """
        self.file_path = file_path
        self.config = self.load_config()

    def load_config(self):
        try:
            with open(self.file_path) as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"No config file found at {self.file_path}. Using default config.")
            return self.default_config()
        except json.JSONDecodeError:
            print(f"Failed to parse JSON from config file at {self.file_path}. Using default config.")
            return self.default_config()

    def default_config(self):
        return {
            'model_paths': {'yolov8n': '/default/path/here'},
            # Add your other default configurations here...
        }




