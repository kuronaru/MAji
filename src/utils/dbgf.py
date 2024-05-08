import logging


def debug_level_config(level):
    logging.basicConfig(level=level)


class DebugPrintf:
    def __init__(self, class_name, debug_level=logging.INFO):
        self.dbgf = logging.getLogger(class_name)
        debug_level_config(debug_level)

    def __call__(self, *args):
        debug_level = logging.DEBUG  # default DEBUG
        msg = args[0]

        if (len(args) == 2):
            debug_level = args[0]
            msg = args[1]

        match (debug_level):
            case logging.CRITICAL: self.dbgf.critical(msg)
            case logging.ERROR: self.dbgf.error(msg)
            case logging.WARNING: self.dbgf.warning(msg)
            case logging.INFO: self.dbgf.info(msg)
            case logging.DEBUG: self.dbgf.debug(msg)
            case _: self.dbgf.debug(msg)
