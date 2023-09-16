import os

from . import devices
from . import keras
from . import strategy

# Utilities ----------------------------------------------------------------------------------------

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
def set_min_log_level(level: str|int):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(level)
