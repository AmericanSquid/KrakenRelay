import yaml
import logging
import os

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
        return {
            'audio': {
                'sample_rate': 48000,
                'chunk_size': 1024,
                'squelch_threshold': -40,
                'highpass_enabled': True,
                'highpass_cutoff': 300,
                'limiter_enabled': True,
                'limiter_threshold': 0.85,
                'dual_output': False,
                'output_device_2': None
            },
            'repeater': {
                'tail_time': 2.0,
                'anti_kerchunk_time': 0,
                'carrier_delay': 0,
                'courtesy_tone_enabled': True
            },
            'identification': {
                'interval_minutes': 10,
                'cw_enabled': True,
                'callsign': "KR4KEN",
                'cw_pitch': 523,
                'cw_wpm': 20,
                'cw_volume': 0.5
            },
            'tot': {
                'tot_enabled': True,
                'tot_time': 180,
                'tot_lockout_enabled': True,
                'tot_lockout_time': 5,
                'tot_tone_freq': 1200
            },
            "ptt": {
                "dual_ptt": True,
                "primary": {
                    "mode": "CM108",
                    "device_path": "/dev/hidraw0",
                    "gpio_pin": "3",
                },
                "secondary": {
                    "mode": "CM108",
                    "gpio_pin": "3",
                    "device_path": "/dev/ttyUSB0",
                }
            }
        }
