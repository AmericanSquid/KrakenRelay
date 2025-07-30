from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                           QLabel, QPushButton, QComboBox, QSlider, QGroupBox,
                           QCheckBox, QTabWidget, QProgressBar, QLineEdit, QSpinBox,
                           QDoubleSpinBox, QSizePolicy)
from PyQt5.QtCore import Qt, QTimer
import numpy as np
from dialogs.mumble import MumbleDialog
from dialogs.ptt import PTTDialog

class RepeaterUI(QMainWindow):
    def __init__(self, config, audio_manager):
        super().__init__()
        self.config = config
        self.audio_manager = audio_manager
        self.controller = None
        self.device_indices = {'input': [], 'output': []}
        
        self.init_ui()
        self.setup_timer()
        
    def init_ui(self):
        self.setWindowTitle('KrakenRelay - Repeater Controller')
        self.setGeometry(100, 100, 600, 800)
        
        # Apply dark theme
        self.setStyleSheet("""
        QMainWindow, QWidget {
            background-color: #23272e;
            color: #ffffff;
            font-family: "Inter", "Segoe UI", "Roboto", sans-serif;
        }
        QGroupBox {
            border: 1px solid #29243c;
            border-radius: 8px;
            margin-top: 1em;
            padding-top: 0.5em;
        }
        QGroupBox::title {
            color: #a259ff;
            font-weight: 600;
        }
        QPushButton {
            background-color: #29243c;
            border: 1.5px solid #37fff8;
            border-radius: 5px;
            color: #ffffff;
            padding: 8px;
            font-weight: 500;
        }
        QPushButton:hover {
            background-color: #20222b;
            border: 1.5px solid #37fff8;
        }
        QPushButton:pressed {
            background-color: #20222b;
            border: 1.5px solid #a259ff;
        }
        QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {
            background-color: #29243c;
            border: 1px solid #29243c;
            border-radius: 3px;
            color: #ffffff;
            padding: 4px;
        }
        QSlider::handle {
            background-color: #a259ff;
            border-radius: 3px;
            width: 18px;
            height: 18px;
            margin: -8px 0;
        }
        QSlider::groove:horizontal {
            background-color: #29243c;
            height: 6px;
            border-radius: 4px;
        }
        QTabWidget::pane {
            border: 1.5px solid #29243c;
            border-radius: 8px;
        }
        QTabBar::tab {
            background-color: #29243c;
            color: #b0b4ba;
            padding: 8px 18px;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
            margin-right: 3px;
            font-weight: 500;
        }
        QTabBar::tab:selected {
            background-color: #22182c;
            color: #ffffff;
            border-bottom: 3px solid #a259ff;
        }
        QTabBar::tab:hover {
            background-color: #29243c;
            color: #ff71ce;
        }
        """)

        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Create tab widget
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        
        # Main Tab
        main_tab = QWidget()
        main_layout = QVBoxLayout(main_tab)
        
        # Audio Device Selection
        devices_group = QGroupBox("Audio Devices")
        devices_layout = QVBoxLayout()
        
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("Input Device:"))
        self.input_combo = QComboBox()
        input_devices = []
        for i in range(self.audio_manager.pa.get_device_count()):
            device_info = self.audio_manager.pa.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:
                input_devices.append(device_info['name'])
                self.device_indices['input'].append(i)
        self.input_combo.addItems(input_devices)
        input_layout.addWidget(self.input_combo)
        devices_layout.addLayout(input_layout)
        
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Output Device:"))
        self.output_combo = QComboBox()
        output_devices = []
        for i in range(self.audio_manager.pa.get_device_count()):
            device_info = self.audio_manager.pa.get_device_info_by_index(i)
            if device_info['maxOutputChannels'] > 0:
                output_devices.append(device_info['name'])
                self.device_indices['output'].append(i)
        self.output_combo.addItems(output_devices)
        output_layout.addWidget(self.output_combo)
        devices_layout.addLayout(output_layout)
        
        devices_group.setLayout(devices_layout)
        main_layout.addWidget(devices_group)
        
        # Noise Reduction Controls
        processing_group = QGroupBox("Noise Reduction")
        processing_layout = QVBoxLayout()
        
        # Squelch Control
        squelch_layout = QHBoxLayout()
        squelch_layout.addWidget(QLabel("Squelch:"))
        self.squelch_slider = QSlider(Qt.Horizontal)
        self.squelch_slider.setMinimum(-60)
        self.squelch_slider.setMaximum(-20)
        self.squelch_slider.setValue(self.config.config['audio']['squelch_threshold'])
        squelch_layout.addWidget(self.squelch_slider)
        self.squelch_value = QLabel(f"{self.squelch_slider.value()} dB")
        squelch_layout.addWidget(self.squelch_value)
        processing_layout.addLayout(squelch_layout)
        
        # High-pass Filter Controls
        highpass_layout = QHBoxLayout()
        self.highpass_enabled = QCheckBox("High-pass Filter")
        self.highpass_enabled.setChecked(self.config.config['audio']['highpass_enabled'])
        highpass_layout.addWidget(self.highpass_enabled)
        
        self.highpass_cutoff = QSlider(Qt.Horizontal)
        self.highpass_cutoff.setRange(100, 1000)
        self.highpass_cutoff.setValue(self.config.config['audio']['highpass_cutoff'])
        highpass_layout.addWidget(self.highpass_cutoff)
        self.highpass_value = QLabel(f"{self.highpass_cutoff.value()} Hz")
        highpass_layout.addWidget(self.highpass_value)
        processing_layout.addLayout(highpass_layout)
        
        # Noise Gate Controls
        noise_gate_layout = QHBoxLayout()
        self.noise_gate_enabled = QCheckBox("Noise Gate")
        self.noise_gate_enabled.setChecked(self.config.config['audio']['noise_gate_enabled'])
        noise_gate_layout.addWidget(self.noise_gate_enabled)
        
        self.noise_gate_threshold = QSlider(Qt.Horizontal)
        self.noise_gate_threshold.setRange(0, 2000)
        self.noise_gate_threshold.setValue(self.config.config['audio']['noise_gate_threshold'])
        noise_gate_layout.addWidget(self.noise_gate_threshold)
        self.noise_gate_value = QLabel(f"{self.noise_gate_threshold.value()}")
        noise_gate_layout.addWidget(self.noise_gate_value)
        processing_layout.addLayout(noise_gate_layout)
        
        processing_group.setLayout(processing_layout)
        main_layout.addWidget(processing_group)
        
        # Audio Meter
        self.setup_audio_meter(main_layout)

        # Mumble LED
        self.mumble_led = QLabel("⚪ Mumble")   # instead of QtWidgets.QLabel
        self.statusBar().addPermanentWidget(self.mumble_led)

        timer = QTimer(self, interval=1000)
        timer.timeout.connect(self._update_mumble_led)
        timer.start()
        
        self.tabs.addTab(main_tab, "Main")
        
        # Settings Tab
        self.setup_settings_tab()
        
        # Control Buttons
        buttons_layout = QHBoxLayout()
        
        self.debug_button = QPushButton("Debug Audio")
        self.debug_button.clicked.connect(self.debug_audio)
        buttons_layout.addWidget(self.debug_button)
        
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_repeater)
        buttons_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_repeater)
        self.stop_button.setEnabled(False)
        buttons_layout.addWidget(self.stop_button)
        
        self.cw_id_button = QPushButton("Send CW ID")
        self.cw_id_button.clicked.connect(self.send_manual_id)
        buttons_layout.addWidget(self.cw_id_button)
        
        layout.addLayout(buttons_layout)
        
        # Attribution text
        attribution = QLabel("KrakenRelay - Open Source Repeater Controller by American Squid/K3AYV")
        attribution.setStyleSheet("""
            QLabel {
                color: #00ff00;
                font-size: 8pt;
                padding: 5px;
            }
        """)
        attribution.setAlignment(Qt.AlignRight)
        layout.addWidget(attribution)
        
        # Connect control signals
        self.squelch_slider.valueChanged.connect(self.update_squelch)
        self.highpass_enabled.stateChanged.connect(self.update_highpass)
        self.highpass_cutoff.valueChanged.connect(self.update_highpass_cutoff)
        self.noise_gate_enabled.stateChanged.connect(self.update_noise_gate)
        self.noise_gate_threshold.valueChanged.connect(self.update_noise_gate_threshold)

    def setup_audio_meter(self, main_layout):
        meter_group = QGroupBox("Audio Level")
        meter_layout = QVBoxLayout()
        
        self.level_label = QLabel("-60 dB")
        self.level_label.setAlignment(Qt.AlignCenter)
        meter_layout.addWidget(self.level_label)
        
        self.level_bar = QProgressBar()
        self.level_bar.setMinimum(-60)
        self.level_bar.setMaximum(0)
        self.level_bar.setOrientation(Qt.Vertical)
        self.level_bar.setTextVisible(False)
        self.level_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid grey;
                background-color: #222222;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:1, x2:0, y2:0,
                                                stop:0 #00ff00,
                                                stop:0.8 #ffff00,
                                                stop:1 #ff0000);
            }
        """)
        meter_layout.addWidget(self.level_bar)
        
        meter_group.setLayout(meter_layout)
        main_layout.addWidget(meter_group)

    def _update_mumble_led(self):
        if self.controller and self.controller.mumble:
            self.mumble_led.setText("🟢 Mumble Online")
        else:
            self.mumble_led.setText("⚪ Mumble Offline")

    def setup_settings_tab(self):
        settings_tab = QWidget()
        settings_layout = QVBoxLayout(settings_tab)
        
        # Repeater Settings
        repeater_group = QGroupBox("Repeater Settings")
        repeater_layout = QVBoxLayout()

        # PL Tone dropdown
        pl_layout = QHBoxLayout()
        pl_layout.addWidget(QLabel("PL Tone:"))
        self.pl_combo = QComboBox()
        ctcss_tones = [67.0, 71.9, 74.4, 77.0, 79.7, 82.5, 85.4, 88.5, 91.5, 94.8,
                       97.4, 100.0, 103.5, 107.2, 110.9, 114.8, 118.8, 123.0, 127.3,
                       131.8, 136.5, 141.3, 146.2, 151.4, 156.7, 162.2, 167.9, 173.8,
                       179.9, 186.2, 192.8, 203.5, 210.7, 218.1, 225.7, 233.6, 241.8, 250.3]
        self.pl_combo.addItems([f"{tone:.1f} Hz" for tone in ctcss_tones])
        current_pl = self.config.config['repeater']['pl_tone_freq']
        index = ctcss_tones.index(current_pl) if current_pl in ctcss_tones else 0
        self.pl_combo.setCurrentIndex(index)
        pl_layout.addWidget(self.pl_combo)
        repeater_layout.addLayout(pl_layout)

        # Precise control sliders
        self.add_precise_control(repeater_layout, "Hang Time (s)", "repeater.tail_time", 0, 10, 0.1)
        self.add_precise_control(repeater_layout, "Anti-Kerchunk (s)", "repeater.anti_kerchunk_time", 0, 5, 0.1)
        self.add_precise_control(repeater_layout, "Carrier Delay (s)", "repeater.carrier_delay", 0, 2, 0.1)

        # Checkbox controls
        self.courtesy_enabled = QCheckBox("Enable Courtesy Tone")
        self.courtesy_enabled.setChecked(self.config.config['repeater']['courtesy_tone_enabled'])
        repeater_layout.addWidget(self.courtesy_enabled)

        repeater_group.setLayout(repeater_layout)
        settings_layout.addWidget(repeater_group)
        # ID Settings
        id_group = QGroupBox("ID Settings")
        id_layout = QVBoxLayout()

        # Callsign input
        callsign_layout = QHBoxLayout()
        callsign_layout.addWidget(QLabel("Callsign:"))
        self.callsign_input = QLineEdit()
        self.callsign_input.setText(self.config.config['repeater']['callsign'])
        callsign_layout.addWidget(self.callsign_input)
        id_layout.addLayout(callsign_layout)

        # ID interval spinbox
        interval_layout = QHBoxLayout()
        interval_layout.addWidget(QLabel("ID Interval (min):"))
        self.id_interval = QSpinBox()
        self.id_interval.setRange(1, 60)
        self.id_interval.setValue(self.config.config['identification']['interval_minutes'])
        interval_layout.addWidget(self.id_interval)
        id_layout.addLayout(interval_layout)

        # CW settings
        self.cw_enabled = QCheckBox("Enable CW ID")
        self.cw_enabled.setChecked(self.config.config['identification']['cw_enabled'])
        id_layout.addWidget(self.cw_enabled)

        cw_speed_layout = QHBoxLayout()
        cw_speed_layout.addWidget(QLabel("CW Speed (WPM):"))
        self.cw_speed = QSpinBox()
        self.cw_speed.setRange(5, 30)
        self.cw_speed.setValue(self.config.config['repeater']['cw_wpm'])
        cw_speed_layout.addWidget(self.cw_speed)
        id_layout.addLayout(cw_speed_layout)

        cw_pitch_layout = QHBoxLayout()
        cw_pitch_layout.addWidget(QLabel("CW Pitch (Hz):"))
        self.cw_pitch = QSpinBox()
        self.cw_pitch.setRange(400, 1200)
        self.cw_pitch.setValue(self.config.config['repeater']['cw_pitch'])
        cw_pitch_layout.addWidget(self.cw_pitch)
        id_layout.addLayout(cw_pitch_layout)

        id_group.setLayout(id_layout)
        settings_layout.addWidget(id_group)
       
        # TOT settings
        tot_group = QGroupBox("Timeout Timer (TOT)")
        tot_layout = QVBoxLayout()
        
        # Enable/Disable TOT
        self.tot_enabled = QCheckBox("Enable TOT")
        self.tot_enabled.setChecked(self.config.config['tot']['tot_enabled'])
        tot_layout.addWidget(self.tot_enabled)

        # TOT Duration
        tot_time_layout = QHBoxLayout()
        tot_time_layout.addWidget(QLabel("Duration (s):"))
        self.tot_time = QDoubleSpinBox()
        self.tot_time.setRange(1, 600)
        self.tot_time.setSuffix(" s")
        self.tot_time.setValue(self.config.config['tot']['tot_time'])
        tot_time_layout.addWidget(self.tot_time)
        tot_layout.addLayout(tot_time_layout)

        # TOT Tone Pitch
        tot_freq_layout = QHBoxLayout()
        tot_freq_layout.addWidget(QLabel("TOT Tone Pitch (Hz):"))
        self.tot_tone_freq = QSpinBox()
        self.tot_tone_freq.setRange(300, 2000)
        self.tot_tone_freq.setValue(self.config.config['tot']['tot_tone_freq'])
        tot_freq_layout.addWidget(self.tot_tone_freq)
        tot_layout.addLayout(tot_freq_layout)

        # Lockout after TOT
        self.tot_lockout = QCheckBox("Enable Lockout")
        self.tot_lockout.setChecked(self.config.config['tot']['tot_lockout_enabled'])
        tot_layout.addWidget(self.tot_lockout)

        # Lockout Duration
        lockout_time_layout = QHBoxLayout()
        lockout_time_layout.addWidget(QLabel("Lockout (s):"))
        self.tot_lockout_time = QDoubleSpinBox()
        self.tot_lockout_time.setRange(0, 300)
        self.tot_lockout_time.setSuffix(" s")
        self.tot_lockout_time.setValue(self.config.config['tot']['tot_lockout_time'])
        lockout_time_layout.addWidget(self.tot_lockout_time)
        tot_layout.addLayout(lockout_time_layout)

        tot_group.setLayout(tot_layout)
        settings_layout.addWidget(tot_group)

        btn_row = QHBoxLayout()

        mumble_btn = QPushButton("Mumble Settings")
        mumble_btn.clicked.connect(self.open_mumble_settings)
        mumble_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        btn_row.addWidget(mumble_btn)

        ptt_btn = QPushButton("PTT Settings")
        ptt_btn.clicked.connect(self.open_ptt_settings)
        ptt_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        btn_row.addWidget(ptt_btn)

        settings_layout.addLayout(btn_row)

        # Connect all settings signals
        self.pl_combo.currentTextChanged.connect(self.update_pl_tone)
        self.courtesy_enabled.stateChanged.connect(self.update_courtesy_tone)
        self.callsign_input.textChanged.connect(self.update_callsign)
        self.id_interval.valueChanged.connect(self.update_id_interval)
        self.cw_enabled.stateChanged.connect(self.update_cw_enabled)
        self.cw_speed.valueChanged.connect(self.update_cw_speed)
        self.cw_pitch.valueChanged.connect(self.update_cw_pitch)
        self.tot_enabled.stateChanged.connect(self.update_tot_enabled)
        self.tot_time.valueChanged.connect(self.update_tot_time)
        self.tot_lockout.stateChanged.connect(self.update_tot_lockout)
        self.tot_lockout_time.valueChanged.connect(self.update_tot_lockout_time)
        self.tot_tone_freq.valueChanged.connect(self.update_tot_tone_freq)

        self.tabs.addTab(settings_tab, "Settings")

    def open_mumble_settings(self):
        cfg = self.config.config['mumble']          # live dict from ConfigManager
        dlg = MumbleDialog(cfg, self)               # modal; class in step 2
        if dlg.exec_():                             # OK pressed
            self.config.config['mumble'] = dlg.result_config()
            self.config.save_config()                      # write YAML
        if self.controller is not None:
            self.controller.reload_mumble_link()

    def open_ptt_settings(self):
        cfg = self.config.config['ptt']  # existing dictionary from config
        dlg = PTTDialog(cfg, self)
        if dlg.exec_():
            self.config.config['ptt'] = dlg.result_config()
            self.config.save_config()
#            if self.controller:
#                self.controller.reload_ptt_config()

    def add_precise_control(self, layout, label, config_path, min_val, max_val, step):
        control_layout = QHBoxLayout()
        control_layout.addWidget(QLabel(label))
        
        slider = QSlider(Qt.Horizontal)
        slider.setRange(int(min_val * 10), int(max_val * 10))
        
        spinbox = QDoubleSpinBox()
        spinbox.setRange(min_val, max_val)
        spinbox.setSingleStep(step)
        
        keys = config_path.split('.')
        value = self.config.config
        for key in keys:
            value = value[key]
        
        slider.setValue(int(value * 10))
        spinbox.setValue(value)
        
        control_layout.addWidget(slider)
        control_layout.addWidget(spinbox)
        
        def update_value():
            value = spinbox.value()
            slider.setValue(int(value * 10))
            current = self.config.config
            for key in keys[:-1]:
                current = current[key]
            current[keys[-1]] = value
            self.config.save_config()
        
        slider.valueChanged.connect(lambda v: spinbox.setValue(v / 10))
        spinbox.valueChanged.connect(update_value)
        
        layout.addLayout(control_layout)

    def setup_timer(self):
        self.meter_timer = QTimer()
        self.meter_timer.timeout.connect(self.update_meter)
        self.meter_timer.start(50)

    def update_meter(self):
        if self.controller:
            level = self.controller.current_rms
            if level > 0:
                db_level = 20 * np.log10(level / 32767)
            else:
                db_level = -60
            self.level_bar.setValue(int(db_level))
            self.level_label.setText(f"{db_level:.1f} dB")
    def update_squelch(self):
        value = self.squelch_slider.value()
        self.config.config['audio']['squelch_threshold'] = value
        self.squelch_value.setText(f"{value} dB")
        self.config.save_config()
        
    def update_highpass(self):
        enabled = self.highpass_enabled.isChecked()
        self.config.config['audio']['highpass_enabled'] = enabled
        self.highpass_cutoff.setEnabled(enabled)
        self.config.save_config()
        
    def update_highpass_cutoff(self):
        value = self.highpass_cutoff.value()
        self.config.config['audio']['highpass_cutoff'] = value
        self.highpass_value.setText(f"{value} Hz")
        self.config.save_config()
        
    def update_noise_gate(self):
        enabled = self.noise_gate_enabled.isChecked()
        self.config.config['audio']['noise_gate_enabled'] = enabled
        self.noise_gate_threshold.setEnabled(enabled)
        self.config.save_config()
        
    def update_noise_gate_threshold(self):
        value = self.noise_gate_threshold.value()
        self.config.config['audio']['noise_gate_threshold'] = value
        self.noise_gate_value.setText(str(value))
        self.config.save_config()

    def update_pl_tone(self, text):
        tone = float(text.split()[0])
        self.config.config['repeater']['pl_tone_freq'] = tone
        self.config.save_config()
    
    def update_courtesy_tone(self):
        enabled = self.courtesy_enabled.isChecked()
        self.config.config['repeater']['courtesy_tone_enabled'] = enabled
        self.config.save_config()
    
    def update_callsign(self):
        callsign = self.callsign_input.text().upper()
        self.config.config['repeater']['callsign'] = callsign
        self.config.save_config()
    
    def update_id_interval(self):
        interval = self.id_interval.value()
        self.config.config['identification']['interval_minutes'] = interval
        self.config.save_config()
    
    def update_cw_enabled(self):
        enabled = self.cw_enabled.isChecked()
        self.config.config['identification']['cw_enabled'] = enabled
        self.config.save_config()
    
    def update_cw_speed(self):
        speed = self.cw_speed.value()
        self.config.config['repeater']['cw_wpm'] = speed
        self.config.save_config()
    
    def update_cw_pitch(self):
        pitch = self.cw_pitch.value()
        self.config.config['repeater']['cw_pitch'] = pitch
        self.config.save_config()
        
    def debug_audio(self):
        self.audio_manager.verify_audio_chain()

    def update_tot_enabled(self, state):
        self.config.config['tot']['tot_enabled'] = bool(state)
        self.config.save_config()
    
    def update_tot_time(self, value):
        self.config.config['tot']['tot_time'] = value
        self.config.save_config()

    def update_tot_tone_freq (self, value):
        self.config.config['tot']['tot_tone_freq'] = value
        self.config.save_config()

    def update_tot_lockout(self, state):
        self.config.config['tot']['tot_lockout_enabled'] = bool(state)
        self.config.save_config()

    def update_tot_lockout_time(self, value):
        self.config.config['tot']['tot_lockout_time'] = value
        self.config.save_config()
        
    def start_repeater(self):
        from repeater_core import RepeaterController
        input_idx = self.device_indices['input'][self.input_combo.currentIndex()]
        output_idx = self.device_indices['output'][self.output_combo.currentIndex()]
        
        self.controller = RepeaterController(
            input_idx,
            output_idx,
            self.config,
            self.audio_manager
        )
        self.controller.start()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        
    def stop_repeater(self):
        if self.controller:
            self.controller.cleanup()
            self.controller = None
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        
    def send_manual_id(self):
        if self.controller:
            self.controller.send_id()
