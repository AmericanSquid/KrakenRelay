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
                'noise_gate_enabled': True,
                'noise_gate_threshold': 500
            },
            'repeater': {
                'pl_tone_freq': 162.2,
                'pl_threshold': 0.1,
                'tail_time': 3.0,
                'anti_kerchunk_time': 1.0,
                'carrier_delay': 0.25,
                'courtesy_tone_enabled': True,
                'cw_wpm': 20,
                'cw_pitch': 800,
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
            }
            
        }
