import pyaudio
import logging
from runtime.logging_utils import debug_enabled

class AudioDeviceManager:
    def __init__(self):
        self.pa = pyaudio.PyAudio()
        logging.info("Audio manager initialized")

    def _iter_devices(self):
        pa = self.pa
        for i in range(pa.get_device_count()):
            yield pa.get_device_info_by_index(i)

    def list_devices(self):
        pa = self.pa
        devices = []
        
        for device_info in self._iter_devices():
            devices.append(device_info)
            if debug_enabled():
                logging.debug(f"Found device: {device_info['name']}")
        return devices

    def get_input_devices(self):
        pa = self.pa
        devices = []
        
        for device_info in self._iter_devices():
            if device_info['maxInputChannels'] > 0:
                devices.append(device_info['name'])
        return devices

    def get_output_devices(self):
        pa = self.pa
        devices = []
        
        for device_info in self._iter_devices():
            if device_info['maxOutputChannels'] > 0:
                devices.append(device_info['name'])
        return devices

    def find_device_by_name(self, name):
        devices = self.list_devices()
        for device in devices:
            if name.lower() in device['name'].lower():
                logging.info(f"Found matching device: {device['name']}")
                return device['index']
        logging.warning(f"No device found matching name: {name}")
        return None

    def create_input_stream(self, device_index, format=pyaudio.paInt16, rate=48000, chunk=1024):
        pa = self.pa
        try:
            device_info = pa.get_device_info_by_index(device_index)
            
            # Use the same rate for both input and output
            actual_rate = int(rate)  # Force consistent sample rate
            
            stream = pa.open(
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
        pa = self.pa
        try:
            # Use same rate as input
            actual_rate = int(rate)
            
            stream = pa.open(
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
        pa = self.pa
        try:
            pa.terminate()
            logging.info("[AudioManager] PortAudio engine terminated.")
        except Exception as e:
            logging.warning(f"[AudioManager] Failed to terminate PortAudio: {e}")

class AudioDeviceError(Exception):
    pass
