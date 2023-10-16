
from typing import Optional
from ...context import Context, context as get_context
from ....lazy import tensorflow as tf

class ContextStoppingCallback(tf.keras.callbacks.Callback):
    def __init__(self, context: Optional[Context] = None):
        super().__init__()
        self._context = context if context else get_context()

    def on_epoch_end(self, epoch, logs=None):
        if not self._context.is_running:
            self.model.stop_training = True
