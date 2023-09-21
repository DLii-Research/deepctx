from ..utils.lazyloading import lazy_wrapper

# Slurm Integration
@lazy_wrapper
def __import(): # type: ignore
    del globals()["slurm"]
    from . import slurm
    return slurm # type: ignore
slurm = __import # Fix IDE highlighting

# Tensorflow Integration
@lazy_wrapper
def __import():
    del globals()["tensorflow"]
    from . import tensorflow
    return tensorflow # type: ignore
tensorflow = __import # Fix IDE highlighting
