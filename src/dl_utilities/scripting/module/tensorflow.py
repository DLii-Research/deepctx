import os
from ...lazy import tensorflow as tf
from .. import ArgumentParser, tensorflow
from ..context import Context, ContextModule

class Tensorflow(ContextModule):

    NAME = "Tensorflow"

    def __init__(self, context: Context):
        super().__init__(context)
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    def min_log_level(self, level: str|int) -> "Tensorflow":
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(level)
        return self
