from .initialize import Initialization
from types import SimpleNamespace

def RepeaterController(input_device, output_device, config, audio_manager):
    controller = SimpleNamespace()
    Initialization(controller).run(
        input_device,
        output_device,
        config,
        audio_manager
    )
    return controller
