import os
import glob
import shutil
import subprocess
import time
from typing import Optional, Dict

# ---------- public API ----------

class SystemStats:
    def __init__(self, cache_interval: int = 5):
        self.cache_interval = cache_interval
        self.last_update = 0.0
        self._cache: Dict[str, Optional[float]] = {}

    def get(self) -> Dict[str, Optional[float]]:
        now = time.time()
        if now - self.last_update < self.cache_interval:
            return self._cache

        self._cache = {
            "cpu_temp": get_cpu_temp(),
            "nvme_temp": get_nvme_temp(),
            "uptime": get_uptime_seconds(),
            "load": get_load_avg(),
            "mumble": mumble_status(),
        }

        self.last_update = now
        return self._cache
# --- Primitives --- #
def _parse_sensors_output(match_token, index):
    try:
        out = subprocess.check_output(["sensors"], text=True, timeout=0.5)
        for line in out.splitlines():
            if match_token in line:
                return float(
                    line.split()[index]
                    .replace("+", "")
                    .replace("°C", "")
                )
    except Exception:
        pass
    return None

def _read_millideg(path):
    try:
        with open(path) as f:
            return int(f.read()) / 1000.0
    except Exception:
        return None

# --- Helpers --- #

def get_uptime_seconds() -> Optional[int]:
    try:
        with open("/proc/uptime") as f:
            return int(float(f.read().split()[0]))
    except Exception:
        return None


def get_load_avg() -> Optional[float]:
    try:
        return os.getloadavg()[0]  # 1-minute load
    except Exception:
        return None

# --- CPU Temp --- #

def get_cpu_temp() -> Optional[float]:
    if shutil.which("vcgencmd"):
        temp = _cpu_temp_vcgencmd()
        if temp is not None:
            return temp

    temp = _cpu_temp_sysfs()
    if temp is not None:
        return temp

    return _cpu_temp_sensors()

def _cpu_temp_vcgencmd() -> Optional[float]:
    try:
        out = subprocess.check_output(
            ["vcgencmd", "measure_temp"],
            text=True,
            timeout=0.5
        )
        # temp=65.2'C
        return float(out.split("=")[1].replace("'C", ""))
    except Exception:
        return None

def _cpu_temp_sysfs() -> Optional[float]:
    try:
        for zone in glob.glob("/sys/class/thermal/thermal_zone*"):
            with open(f"{zone}/type") as f:
                t = f.read().strip()

            if t in ("x86_pkg_temp", "cpu-thermal", "soc-thermal"):
                return _read_millideg(f"{zone}/type")
    except Exception:
        pass

    return None

def _cpu_temp_sensors():
    return _parse_sensors_output("Package id 0:", 3)
# --- nVME Temp --- #

def get_nvme_temp() -> Optional[float]:
    temp = _nvme_temp_sysfs()
    if temp is not None:
        return temp



    return _nvme_temp_sensors()

def _nvme_temp_sysfs() -> Optional[float]:
    try:
        for path in glob.glob("/sys/class/nvme/nvme*/device/hwmon/hwmon*/temp1_input"):
            return _read_millideg(path)
    except Exception:
        pass

    return None

def _nvme_temp_sensors():
    return _parse_sensors_output("Composite:", 1)

def mumble_status():
    try:
        result = subprocess.run(
            ["systemctl", "is-active", "mumble-cli"],
            capture_output=True,
            text=True
        )
        return result.stdout.strip() == "active" 
    except Exception:
        return False

if __name__ == "__main__":
    stats = SystemStats()
    data = stats.get()

    print("=== KrakenRelay System Stats ===")
    for key, value in data.items():
        print(f"{key:12}: {value}")

