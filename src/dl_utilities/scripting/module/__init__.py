from ...utils.lazyloading import lazy_wrapper

# Lazy Module Loading ------------------------------------------------------------------------------

@lazy_wrapper
def Rng(): # type: ignore
    from .rng import Rng
    return Rng

@lazy_wrapper
def Tensorflow():
    from .tensorflow import Tensorflow
    return Tensorflow

@lazy_wrapper
def Train(): # type: ignore
    from .train import Train
    return Train

@lazy_wrapper
def Wandb(): # type: ignore
    from .wandb import Wandb
    return Wandb
