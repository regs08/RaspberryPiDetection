from RaspberryPiDetection1.Config.config_loader import ConfigLoader


json_path = "/Users/cole/PycharmProjects/kivyTutorial/repo/App/Config/config.json"
default_config = ConfigLoader(json_path).load_config()