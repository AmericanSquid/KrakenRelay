import yaml
import logging
import os

class ConfigManager:
    def __init__(self, config_path='config.yaml'):
        self.config_path = config_path
        self.config = self.load_config()
        if self.config is None:
            self.config = self.get_default_config()
        
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
                'input_gain': 0,
                'output_gain': 0,
                'squelch_threshold': -40,
                'highpass_enabled': True,
                'highpass_cutoff': 300,
                'noise_gate_enabled': False,
                'noise_gate_threshold': 500
            },
            'repeater': {
                'pl_tone_freq': 162.2,
                'pl_threshold': 0.1,
                'tail_time': 2.0,
                'anti_kerchunk_time': 0,
                'carrier_delay': 0,
                'courtesy_tone_enabled': True,
                'cw_wpm': 18,
                'cw_pitch': 700,
                'callsign': 'K3AYV'
            },
            'identification': {
                'interval_minutes': 10,
                'cw_enabled': True
            },
            'tot': {
                'tot_enabled': True,
                'tot_time': 180,
                'tot_lockout_enabled': True,
                'tot_lockout_time': 5,
                'tot_tone_freq': 1200
            },
            "mumble": {
                "enabled": False,
                "mode": "link",          # link | voter   (voter stubbed)
                "direction": "bidirectional",     # bidirectional | rf_to_mumble | mumble_to_rf
                "host": "127.0.0.1",
                "port": 64738,
                "user": "KrakenRelay",
                "password": "",
                "channel": "RepeaterLink",
            }, 
            "ptt": {
                "mode": "CM108",
                "device_path": "/dev/hidraw0",
                "gpio_pin": "3",
            }
        }
