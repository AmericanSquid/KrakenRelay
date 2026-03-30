import sys
from audio import list_audio_devices
from runtime import (
    setup_logging,
    parse_args,
    sig_handler
)
from runtime.launch import run_web, run_headless

# -------------#
# Entry-point  #
# -------------#
def main() -> None:
    args = parse_args()
    setup_logging(args.debug)

    if args.list_devices:
        list_audio_devices()
        sys.exit(0)

    if args.headless:
        run_headless(args)
    else:
        run_web(args)

if __name__ == "__main__":
    main()
