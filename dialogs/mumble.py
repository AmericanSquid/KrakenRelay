from PyQt5 import QtWidgets, QtCore

class MumbleDialog(QtWidgets.QDialog):
    def __init__(self, cfg, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Mumble Settings")
        form = QtWidgets.QFormLayout(self)

        # widgets
        self.enable    = QtWidgets.QCheckBox("Enable")
        self.enable.setChecked(cfg['enabled'])

        self.mode_group = QtWidgets.QButtonGroup(self)
        self.mode_layout = QtWidgets.QHBoxLayout()
        self.mode_buttons = {}
        for i, mode in enumerate(["Link", "Voter"]):
            btn = QtWidgets.QPushButton(mode)
            btn.setCheckable(True)
            btn.setMinimumWidth(80)
            if cfg['mode'] == mode:
                btn.setChecked(True)
            self.mode_group.addButton(btn, i)
            self.mode_layout.addWidget(btn)
            self.mode_buttons[mode] = btn
	
            btn.setStyleSheet("""
                QPushButton {
                    border: 1px solid #505050;
                    border-radius: 0px;
                    background: #353535;
                    color: #fff;
                    padding: 6px 18px;
                    font-weight: normal;
                }
                QPushButton:checked {
                    background: #00ff00;
                    color: #232323;
                    font-weight: bold;
                }
                QPushButton:hover { background: #505050; }
            """)

        self.direction_group = QtWidgets.QButtonGroup(self)
        self.direction_layout = QtWidgets.QHBoxLayout()
        self.direction_buttons = {}
        for i, direction in enumerate(["Bidirectional", "RF-to-Mumble", "Mumble-to-RF"]):
            btn = QtWidgets.QPushButton(direction)
            btn.setCheckable(True)
            btn.setMinimumWidth(110)
            if cfg['direction'] == direction:
                btn.setChecked(True)
            self.direction_group.addButton(btn, i)
            self.direction_layout.addWidget(btn)
            self.direction_buttons[direction] = btn
            btn.setStyleSheet("""
                QPushButton {
                    border: 1px solid #505050;
                    border-radius: 0px;
                    background: #353535;
                    color: #fff;
                    padding: 6px 10px;
                    font-weight: normal;
                }
                QPushButton:checked {
                    background: #00ff00;
                    color: #232323;
                    font-weight: bold;
                }
                QPushButton:hover { background: #505050; }
            """)

        self.host      = QtWidgets.QLineEdit(cfg['host'])
        self.port      = QtWidgets.QSpinBox(); self.port.setRange(1, 65535)
        self.port.setValue(cfg['port'])
        self.user      = QtWidgets.QLineEdit(cfg['user'])
        self.password  = QtWidgets.QLineEdit(cfg['password'])
        self.password.setEchoMode(QtWidgets.QLineEdit.Password)
        self.channel   = QtWidgets.QLineEdit(cfg['channel'])

        form.addRow("", self.enable)
        form.addRow("Mode", self.mode_layout)
        form.addRow("Direction", self.direction_layout)
        for lab, w in [("Host", self.host), ("Port", self.port),
                       ("User", self.user), ("Password", self.password),
                       ("Channel", self.channel)]:
            form.addRow(lab, w)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    # helper for the caller
    def result_config(self):
        mode = next((text for text, btn in self.mode_buttons.items() if btn.isChecked()), "Link")
        direction = next((text for text, btn in self.direction_buttons.items() if btn.isChecked()), "Bidirectional")
        # Normalize to match backend expectations
        mode_norm = mode.lower().replace('-', '_').replace(' ', '')
        direction_norm = direction.lower().replace('-', '_').replace(' ', '')
        return dict(
            enabled   = self.enable.isChecked(),
            mode      = mode_norm,
            direction = direction_norm,
            host      = self.host.text(),
            port      = self.port.value(),
            user      = self.user.text(),
            password  = self.password.text(),
            channel   = self.channel.text()
        )

