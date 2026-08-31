from .initialize import Initialization


def build_repeater(
    input_device,
    output_device,
    config,
    audio_manager,
    audit=None,
    publish_services=None,
):
    return Initialization().run(
        input_device,
        output_device,
        config,
        audio_manager,
        audit=audit,
        publish_services=publish_services,
    )
