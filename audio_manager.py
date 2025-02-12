import pyaudio
import numpy as np
import logging
from scipy.signal import butter, lfilter
from collections import deque
import threading

class AudioBuffer:
    def __init__(self, maxlen=10):
        self.buffer = deque(maxlen=maxlen)
        self.lock = threading.Lock()
        
    def write(self, data):
        with self.lock:
            self.buffer.append(data)
            
    def read(self):
        with self.lock:
            return self.buffer.popleft() if self.buffer else None

class AudioDeviceManager:
    def __init__(self):
        self.pa = pyaudio.PyAudio()
        self.tone_detector = ToneDetector()
        self.input_buffer = AudioBuffer()
        self.output_buffer = AudioBuffer()
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

    def calibrate_levels(self):
        samples = []
        for _ in range(30):
            data = self.input_stream.read(1024)
            samples.extend(np.frombuffer(data, dtype=np.int16))
        noise_floor = np.mean(np.abs(samples))
        return noise_floor
    
    def cleanup(self):
        logging.info("Cleaning up audio manager")
        self.pa.terminate()

class ToneDetector:
    def __init__(self, sample_rate=44100, target_freq=141.3, bandwidth=3.0, threshold=0.001, alpha=0.3):
        self.target_freq = target_freq
        self.bandwidth = bandwidth
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.alpha = alpha
        self.smoothed_energy = 0.0

        lowcut = target_freq - bandwidth
        highcut = target_freq + bandwidth
        self.b, self.a = butter(4, [lowcut, highcut], fs=sample_rate, btype='band')
        
        logging.info(f"ToneDetector initialized: target={target_freq}Hz, bandwidth=±{bandwidth}Hz")

    def detect_tone(self, audio_data):
        filtered = lfilter(self.b, self.a, audio_data)
        energy = np.sum(filtered ** 2)
        self.smoothed_energy = self.alpha * energy + (1 - self.alpha) * self.smoothed_energy
        logging.debug(f"ToneDetector: energy={self.smoothed_energy:.6f}")
        return self.smoothed_energy > self.threshold

class AudioDeviceError(Exception):
    pass
