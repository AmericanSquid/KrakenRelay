import logging

#------------------#
# Logging Helpers  #
#------------------#
def setup_logging(debug: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
def _debug_enabled():
    return logging.getLogger().isEnabledFor(logging.DEBUG)

