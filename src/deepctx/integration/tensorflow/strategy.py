import tensorflow as tf

__strategy = None

def auto() -> tf.distribute.Strategy:
    """
    Get the current strategy.
    """
    global __strategy
    if __strategy is None:
        cpus = [gpu.name.split(':', maxsplit=1)[1] for gpu in tf.config.get_visible_devices("CPU")]
        gpus = [cpu.name.split(':', maxsplit=1)[1] for cpu in tf.config.get_visible_devices("GPU")]
        if len(gpus) > 1:
            __strategy = tf.distribute.MirroredStrategy(cpus + gpus)
        elif len(gpus) == 1:
            __strategy = tf.distribute.OneDeviceStrategy(gpus[0])
        else:
            __strategy = tf.distribute.OneDeviceStrategy(tf.config.get_visible_devices("CPU"))
    return __strategy

