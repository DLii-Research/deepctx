from .context import ArgumentParser, context, Context, execute, Job, State

# Module Imports -----------------------------------------------------------------------------------

def rng():
    from .module.rng import Rng
    return Rng

def tensorflow():
    from .module.tensorflow import Tensorflow
    return Tensorflow

def train():
    from .module.train import Train
    return Train

def wandb():
    from .module.wandb import Wandb
    return Wandb
