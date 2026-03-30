import yaml
import logging
import copy
from .template import DEFAULT_CONFIG

class ConfigManager:
    def __init__(self, config_path='config.yaml'):
        self.config_path = config_path
        self.config = self.load_config()
        if self.config is None:
            logging.warning(f"No config found at '{self.config_path}' — generating default.")
            self.config = self.get_default_config()
            self.save_config()
            logging.info(f"Default config saved to '{self.config_path}'")
        else:
            logging.info("Config loaded successfully.")
        
    def load_config(self):
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
                logging.info("Configuration loaded successfully")
                return config
        except Exception as e:
            logging.error(f"Error loading configuration: {e}")
            return self.get_default_config()
    
    def save_config(self):
        try:
            with open(self.config_path, 'w') as file:
                yaml.dump(self.config, file)
                logging.info("Configuration saved successfully")
        except Exception as e:
            logging.error(f"Error saving configuration: {e}")
    
    def get_default_config(self):
        return copy.deepcopy(DEFAULT_CONFIG)
