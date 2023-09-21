from ...utils.lazyloading import lazy_wrapper

# Lazy Module Loading ------------------------------------------------------------------------------

@lazy_wrapper
def Rng(): # type: ignore
    from .rng_module import Rng
    return Rng

@lazy_wrapper
def Tensorflow():
    from .tensorflow_module import Tensorflow
    return Tensorflow

@lazy_wrapper
def Train(): # type: ignore
    from .train_module import Train
    return Train

@lazy_wrapper
def Wandb(): # type: ignore
    from .wandb_module import Wandb
    return Wandb
