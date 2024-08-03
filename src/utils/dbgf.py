import logging

GLOBAL_DBG_LVL = logging.INFO
DATA_PROCESS_DBG_LVL = logging.DEBUG


def debug_level_config(level):
    logging.basicConfig(level=level)


class DebugPrintf:
    def __init__(self, class_name, debug_level=logging.INFO):
        self.logger = logging.getLogger(class_name)
        self.logger.setLevel(debug_level)
        handler = logging.StreamHandler()
        handler.setLevel(debug_level)
        formatter = logging.Formatter("[%(levelname)s]%(name)s: %(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def __call__(self, *args):
        debug_level = logging.DEBUG  # default DEBUG
        msg = args[0]

        if (len(args) == 2):
            debug_level = args[0]
            msg = args[1]

        match (debug_level):
            case logging.CRITICAL: self.logger.critical(msg)
            case logging.ERROR: self.logger.error(msg)
            case logging.WARNING: self.logger.warning(msg)
            case logging.INFO: self.logger.info(msg)
            case logging.DEBUG: self.logger.debug(msg)
            case _: self.logger.debug(msg)
