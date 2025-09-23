import pyaudio
import numpy as np
import logging
from scipy.signal import butter, lfilter
from collections import deque
import threading

class AudioDeviceManager:
    def __init__(self):
        self.pa = pyaudio.PyAudio()
        logging.info("Audio manager initialized")
        
    def list_devices(self):
        devices = []
        for i in range(self.pa.get_device_count()):
            device_info = self.pa.get_device_info_by_index(i)
            devices.append(device_info)
            logging.info(f"Found device: {device_info['name']}")
        return devices

    def get_input_devices(self):
        devices = []
        for i in range(self.pa.get_device_count()):
            device_info = self.pa.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:
                devices.append(device_info['name'])
        return devices

    def get_output_devices(self):
        devices = []
        for i in range(self.pa.get_device_count()):
            device_info = self.pa.get_device_info_by_index(i)
            if device_info['maxOutputChannels'] > 0:
                devices.append(device_info['name'])
        return devices

    
    def verify_audio_chain(self):
        devices = self.list_devices()
        logging.info("Active audio devices:")
        for device in devices:
            logging.info(f"Index {device['index']}: {device['name']}")
            logging.info(f"Max Input Channels: {device['maxInputChannels']}")
            logging.info(f"Max Output Channels: {device['maxOutputChannels']}")
            logging.info(f"Default Sample Rate: {device['defaultSampleRate']}")

    def find_device_by_name(self, name):
        devices = self.list_devices()
        for device in devices:
            if name.lower() in device['name'].lower():
                logging.info(f"Found matching device: {device['name']}")
                return device['index']
        logging.warning(f"No device found matching name: {name}")
        return None
    
    def create_input_stream(self, device_index, format=pyaudio.paInt16, rate=48000, chunk=1024):
        try:
            device_info = self.pa.get_device_info_by_index(device_index)
            channels = int(device_info['maxInputChannels'])
            
            # Use the same rate for both input and output
            actual_rate = int(rate)  # Force consistent sample rate
            
            stream = self.pa.open(
                format=format,
                channels=1,  # Mono input
                rate=actual_rate,  # Use specified rate
                input=True,
                frames_per_buffer=chunk,
                input_device_index=device_index
            )
            logging.info(f"Created input stream: rate={actual_rate}, channels=1")
            return stream
        except Exception as e:
            logging.error(f"Failed to create input stream: {e}")
            raise AudioDeviceError(f"Could not open input device {device_index}")
    
    def create_output_stream(self, device_index, format=pyaudio.paInt16, channels=1, rate=48000, chunk=1024):
        try:
            # Use same rate as input
            actual_rate = int(rate)
            
            stream = self.pa.open(
                format=format,
                channels=channels,
                rate=actual_rate,
                output=True,
                frames_per_buffer=chunk,
                output_device_index=device_index
            )
            logging.info(f"Created output stream: rate={actual_rate}, channels={channels}")
            return stream
        except Exception as e:
            logging.error(f"Failed to create output stream: {e}")
            raise AudioDeviceError(f"Could not open output device {device_index}")

    def cleanup(self):
        logging.info("[AudioManager] Cleaning up audio streams.")

        try:
            if self.input_stream:
                self.input_stream.stop_stream()
                self.input_stream.close()
        except Exception as e:
            logging.warning(f"Failed to stop/close input stream: {e}")

        try:
            if self.output_stream:
                self.output_stream.stop_stream()
                self.output_stream.close()
        except Exception as e:
            logging.warning(f"Failed to stop/close output stream: {e}")

        try:
            if self.output_stream_2:
                self.output_stream_2.stop_stream()
                self.output_stream_2.close()
        except Exception as e:
            logging.warning(f"Failed to stop/close output stream 2: {e}")

        try:
            self.pa.terminate()
            logging.info("[AudioManager] PortAudio engine terminated.")
        except Exception as e:
            logging.warning(f"[AudioManager] Failed to terminate PortAudio: {e}")

class AudioDeviceError(Exception):
    pass
