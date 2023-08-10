import json
import os


def update_paths(repo_path):
    config_file_path = os.path.join(repo_path, 'config.json')
    # get project dir
    project_dir = os.path.dirname(repo_path)
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

    # Write the updated config back to the file
    with open(config_file_path, 'w') as file:
        json.dump(config, file, indent=4)


if __name__ == "__main__":
    repo_path = os.path.abspath(os.path.dirname(__file__))
    update_paths(repo_path)
    print("Config file updated successfully.")
