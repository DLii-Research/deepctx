import tensorflow as tf
from typing import Union

__strategy = None

def auto() -> Union[tf.distribute.OneDeviceStrategy, tf.distribute.MirroredStrategy]:
    """
    Get the current strategy.
    """
    global __strategy
    if __strategy is None:
        cpus = [cpu.name.split(':', maxsplit=1)[1] for cpu in tf.config.get_visible_devices("CPU")]
        gpus = [gpu.name.split(':', maxsplit=1)[1] for gpu in tf.config.get_visible_devices("GPU")]
        if len(gpus) > 1:
            __strategy = tf.distribute.MirroredStrategy(gpus)
        elif len(gpus) == 1:
            __strategy = tf.distribute.OneDeviceStrategy(gpus[0])
        else:
            __strategy = tf.distribute.OneDeviceStrategy(cpus[0])
    return __strategy

