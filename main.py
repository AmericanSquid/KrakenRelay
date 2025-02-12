import sys
import logging
from PyQt5.QtWidgets import QApplication
from config_manager import ConfigManager
from audio_manager import AudioDeviceManager
from ui import RepeaterUI

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def main():
    setup_logging()
    app = QApplication(sys.argv)
    
    config = ConfigManager()
    audio_manager = AudioDeviceManager()
    
    ui = RepeaterUI(config, audio_manager)
    ui.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
