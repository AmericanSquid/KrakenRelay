from PyQt5.QtWidgets import (QDialog, QFormLayout, QLineEdit, QSpinBox,
                             QComboBox, QDialogButtonBox, QPushButton, QHBoxLayout)

class PTTDialog(QDialog):
    def __init__(self, ptt_config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("PTT Configuration")
        self.result = dict(ptt_config)

        layout = QFormLayout(self)

        self.device_path_input = QLineEdit(self.result.get("device_path", ""))
        self.gpio_pin_input = QSpinBox()
        self.gpio_pin_input.setRange(0, 40)
        self.gpio_pin_input.setValue(self.result.get("gpio_pin", 3))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["None", "CM108"])
        self.mode_combo.setCurrentText(self.result.get("mode", "None"))

        layout.addRow("Device Path:", self.device_path_input)
        layout.addRow("GPIO Pin:", self.gpio_pin_input)
        layout.addRow("PTT Mode:", self.mode_combo)

        button_layout = QHBoxLayout()
        self.test_button = QPushButton("Test PTT")
        button_layout.addWidget(self.test_button)
        layout.addRow(button_layout)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def result_config(self):
        return {
            "device_path": self.device_path_input.text(),
            "gpio_pin": self.gpio_pin_input.value(),
            "mode": self.mode_combo.currentText()
        }