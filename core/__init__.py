from .initialize import Initialization


def build_repeater(
    input_device,
    output_device,
    config,
    audio_manager,
    audit=None,
    publish_services=None,
    ptt_manager_factory=None,
):
    return Initialization().run(
        input_device,
        output_device,
        config,
        audio_manager,
        audit=audit,
        publish_services=publish_services,
        ptt_manager_factory=ptt_manager_factory,
    )
