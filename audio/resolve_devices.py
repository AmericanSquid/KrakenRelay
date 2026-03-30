from typing import Optional
from . import AudioDeviceManager

#---------------#
# List Devices  #
#---------------#
def list_audio_devices() -> None:
    am = AudioDeviceManager()
    try:
        devices = am.list_devices()
        print("Available audio devices:")
        for device in devices:
            print(
                f"  {device['index']}: {device['name']} "
                f"(in={device['maxInputChannels']} out={device['maxOutputChannels']})"
            )
    finally:
        am.cleanup()

#----------------------------#
# Device Resolution Helpers  #
#----------------------------#
def resolve_device(arg: Optional[str], am: AudioDeviceManager, direction: str) -> int:
    if arg is None:
        default_info = (
            am.pa.get_default_input_device_info()
            if direction == "input"
            else am.pa.get_default_output_device_info()
        )
        return int(default_info["index"])

    if isinstance(arg, int):
        return arg

    if isinstance(arg, str) and arg.isdigit():
        return int(arg)

    idx = am.find_device_by_name(arg)
    if idx is None:
        raise ValueError(f"No {direction} device matches '{arg}'.")
    return idx
