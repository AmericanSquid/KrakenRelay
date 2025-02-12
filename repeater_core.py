import numpy as np
import time
import threading
import logging
from scipy import signal
from morse_code import MorseCode
from audio_manager import ToneDetector, AudioBuffer

class RepeaterController:
    def __init__(self, input_device, output_device, config, audio_manager):
        self.config = config
        self.audio_manager = audio_manager
        self.input_device = input_device
        self.output_device = output_device
        self.input_stream = None
        self.output_stream = None
        self.running = False
        self.transmitting = False
        self.last_audio_time = time.time()
        self.last_transmission = time.time()
        self.current_rms = 0
        
        # Initialize audio buffers
        self.input_buffer = AudioBuffer()
        self.output_buffer = AudioBuffer()
        
        self.tone_detector = ToneDetector(
            sample_rate=config.config['audio']['sample_rate'],
            target_freq=config.config['repeater']['pl_tone_freq'],
            bandwidth=3.0,
            threshold=config.config['repeater']['pl_threshold'],
            alpha=0.3
        )

        self.morse = MorseCode(
            wpm=config.config['repeater']['cw_wpm'],
            frequency=config.config['repeater']['cw_pitch'],
            sample_rate=config.config['audio']['sample_rate']
        )
        
        self.last_id_time = time.time()
        self.generate_courtesy_tone()
        self.setup_audio_streams()
        logging.info("RepeaterController initialized")

    def check_squelch(self, samples):
        db_level = 20 * np.log10(np.sqrt(np.mean(samples**2)) / 32767)
        return db_level > self.config.config['audio']['squelch_threshold']

    def apply_highpass_filter(self, samples):
        if not self.config.config['audio']['highpass_enabled']:
            return samples
        nyquist = self.config.config['audio']['sample_rate'] / 2
        cutoff = self.config.config['audio']['highpass_cutoff']
        b, a = signal.butter(4, cutoff/nyquist, btype='high')
        return signal.filtfilt(b, a, samples)

    def apply_noise_gate(self, samples):
        if not self.config.config['audio']['noise_gate_enabled']:
            return samples
        threshold = self.config.config['audio']['noise_gate_threshold']
        samples[abs(samples) < threshold] = 0
        return samples

    def start(self):
        self.running = True
        self.audio_thread = threading.Thread(target=self.audio_loop)
        self.audio_thread.start()
        logging.info("Repeater controller started")

    def setup_audio_streams(self):
        try:
            audio_config = self.config.config['audio']
            sample_rate = audio_config['sample_rate']
            chunk_size = audio_config['chunk_size']
            
            logging.info(f"Setting up audio streams with rate: {sample_rate}")
            
            self.input_stream = self.audio_manager.create_input_stream(
                device_index=self.input_device,
                rate=sample_rate,
                chunk=chunk_size
            )
            
            self.output_stream = self.audio_manager.create_output_stream(
                device_index=self.output_device,
                rate=sample_rate,  # Use same rate as input
                chunk=chunk_size
            )
            
            logging.info(f"Audio streams setup complete: rate={sample_rate}, chunk={chunk_size}")
            
        except Exception as e:
            logging.error(f"Failed to setup audio streams: {e}")
            raise

    def generate_courtesy_tone(self):
        sample_rate = self.config.config['audio']['sample_rate']
        duration = 0.1
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        tone1 = np.sin(2 * np.pi * 800 * t)
        tone2 = np.sin(2 * np.pi * 400 * t)
        
        gap = np.zeros(int(sample_rate * 0.05))
        
        self.courtesy_tone = np.concatenate([tone1, gap, tone2])
        
        fade_len = int(sample_rate * 0.01)
        fade = np.linspace(0, 1, fade_len)
        self.courtesy_tone[:fade_len] *= fade
        self.courtesy_tone[-fade_len:] *= fade[::-1]
        
        self.courtesy_tone = (self.courtesy_tone * 32767).astype(np.int16)
        logging.info("Generated courtesy tone")

    def play_courtesy_tone(self):
        if self.transmitting and self.config.config['repeater']['courtesy_tone_enabled']:
            self.output_stream.write(self.courtesy_tone.tobytes())
            logging.info("Played courtesy tone")

    def start_transmission(self):
        if time.time() - self.last_transmission > self.config.config['repeater']['anti_kerchunk_time']:
            self.transmitting = True
            time.sleep(self.config.config['repeater']['carrier_delay'])
            logging.info("Starting transmission")

    def stop_transmission(self):
        if self.config.config['repeater']['courtesy_tone_enabled']:
            time.sleep(0.1)
            self.play_courtesy_tone()
            time.sleep(0.2)
        
        fade_duration = 0.1
        sample_rate = self.config.config['audio']['sample_rate']
        chunk_size = self.config.config['audio']['chunk_size']
        num_fade_chunks = int((fade_duration * sample_rate) // chunk_size)
        
        for i in range(num_fade_chunks):
            silent_chunk = (np.zeros(chunk_size)).astype(np.int16).tobytes()
            self.output_stream.write(silent_chunk)
            time.sleep(chunk_size / sample_rate)
        
        self.transmitting = False
        self.last_transmission = time.time()
        logging.info("Transmission stopped with fade-out")

    def play_audio_chunks(self, audio):
        chunk_size = self.config.config['audio']['chunk_size']
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i + chunk_size]
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
            self.output_stream.write(chunk.tobytes())

    def send_id(self):
        try:
            callsign = self.config.config['repeater']['callsign']
            if self.config.config['identification']['cw_enabled']:
                self.start_transmission()
                time.sleep(0.2)
                cw_audio = self.morse.generate(callsign)
                self.play_audio_chunks(cw_audio)
                time.sleep(0.2)
                self.stop_transmission()
                logging.info(f"Sent CW ID: {callsign}")
        except Exception as e:
            logging.error(f"ID failed: {e}")
        finally:
            self.last_id_time = time.time()

    def audio_loop(self):
        while self.running:
            try:
                self.process_audio()
                if time.time() - self.last_id_time > self.config.config['identification']['interval_minutes'] * 60:
                    self.send_id()
            except Exception as e:
                logging.error(f"Error in audio loop: {e}")

    def calculate_db_level(self, samples):
        rms = np.sqrt(np.mean(samples**2))
        if rms > 0:
            db_level = 20 * np.log10(rms / 32767)
        else:
            db_level = -60  # Set minimum dB level for silence
        return db_level

    def process_audio(self):
        data = self.input_stream.read(self.config.config['audio']['chunk_size'], exception_on_overflow=False)
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
        self.current_rms = self.calculate_db_level(samples)
        
        # Check squelch first
        if self.check_squelch(samples):
            # Process audio chain
            samples = self.apply_noise_gate(samples)
            samples = self.apply_highpass_filter(samples)
            
            # Apply input gain
            samples *= 10 ** (self.config.config['audio']['input_gain'] / 20)
            
            # Update meter with processed audio
            self.current_rms = np.sqrt(np.mean(samples**2))
            
            # Check for PL tone
            if self.tone_detector.detect_tone(samples):
                if not self.transmitting:
                    logging.info("141.3 Hz PL Tone Detected")
                self.handle_transmission(samples)
        else:
            self.current_rms = 0
            if self.transmitting and (time.time() - self.last_audio_time) > self.config.config['repeater']['tail_time']:
                self.stop_transmission()

    def handle_transmission(self, samples):
        if not self.transmitting:
            self.start_transmission()
        
        # Apply output gain
        output_samples = samples * 10 ** (self.config.config['audio']['output_gain'] / 20)
        output_data = output_samples.astype(np.int16).tobytes()
        self.output_stream.write(output_data)
        
        self.last_audio_time = time.time()

    def cleanup(self):
        self.running = False
        if hasattr(self, 'audio_thread'):
            self.audio_thread.join(timeout=1.0)
        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()
        if self.output_stream:
            self.output_stream.stop_stream()
            self.output_stream.close()
