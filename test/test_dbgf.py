import logging

from src.utils.dbgf import DebugPrintf

dbgf = DebugPrintf("class_test", logging.INFO)

dbgf(logging.ERROR, "test dbgf error")
dbgf(logging.INFO, "test dbgf info")
dbgf(logging.DEBUG, "test dbgf debug")
dbgf("test dbgf")
