import json
import os


def update_paths(project_dir):
    config_file_path = os.path.join(project_dir, 'config.json')
    # get project dir
    project_dir = os.path.dirname(project_dir)
    # where models are stored
    model_dir = os.path.join(project_dir, 'Models')

    # Read the existing config
    with open(config_file_path, 'r') as file:
        config = json.load(file)

    # Replace the paths
    for key, value in config['model_paths'].items():
        filename = os.path.basename(value)
        #new_path = value.replace("...", repo_path)
        config['model_paths'][key] = os.path.join(model_dir, filename)
    # updating config path assuming that is named config.json
    config['config_path'] = os.path.join(model_dir, 'config.json')
    # Write the updated config back to the file
    with open(config_file_path, 'w') as file:
        json.dump(config, file, indent=4)


if __name__ == "__main__":
    project_dir = os.path.abspath(os.path.dirname(__file__))
    update_paths(project_dir)
    print("Config file updated successfully.")
