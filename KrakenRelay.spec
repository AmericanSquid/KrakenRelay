# -*- mode: python ; coding: utf-8 -*-

import sys
import os
from PyInstaller.utils.hooks import collect_submodules

# Ensure all required modules are included
hidden_imports = (
    collect_submodules('PyQt5') +
    collect_submodules('numpy') +
    collect_submodules('scipy') +
    collect_submodules('yaml') +
    collect_submodules('pyaudio') +
    collect_submodules('pyqtgraph')
)

# Include necessary files (icons & config)
datas = [
    ('config.yaml', '.'),  # Config file
    ('krakenrelay.ico', '.'),  # Windows icon
    ('krakenrelay.png', '.'),  # Mac/Linux icon
]

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=1,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='KrakenRelay',
    icon='krakenrelay.ico' if sys.platform == "win32" else 'krakenrelay.png',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # Hide console for GUI apps
    disable_windowed_traceback=False,
    argv_emulation=True if sys.platform == "darwin" else False,
    target_arch="x86_64" if sys.platform == "win32" else None,
    codesign_identity="Developer ID Application" if sys.platform == "darwin" else None,
    entitlements_file="entitlements.plist" if sys.platform == "darwin" else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='KrakenRelay',
)
