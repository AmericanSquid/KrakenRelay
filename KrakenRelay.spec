from pathlib import Path

from PyInstaller.config import CONF
from PyInstaller.utils.hooks import collect_submodules


ROOT = Path(CONF["spec"]).__parent


hiddenimports = []
for package in ("audio", "config", "core", "dsp", "plugins", "ptt", "runtime", "tones", "web_ui"):
    hiddenimports.extend(collect_submodules(package))


datas = [
    (str(ROOT / "web_ui" / "templates"), "web_ui/templates"),
    (str(ROOT / "web_ui" / "static"), "web_ui/static"),
    (str(ROOT / "plugins" / "config.yaml"), "plugins"),
    (str(ROOT / "config.yaml.example"), "."),
    (str(ROOT / "audio" / "config.yaml.example"), "audio"),
]


a = Analysis(
    [str(ROOT / "run.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tests", "matplotlib.tests"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="KrakenRelay",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
)
