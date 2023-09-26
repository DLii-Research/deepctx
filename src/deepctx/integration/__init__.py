from ..utils.lazyloading import lazy_wrapper

# Slurm Integration
@lazy_wrapper
def slurm(): # type: ignore
    del globals()["slurm"]
    from . import slurm
    globals()["slurm"] = slurm
    return slurm # type: ignore

# Tensorflow Integration
@lazy_wrapper
def tensorflow():
    del globals()["tensorflow"]
    from . import tensorflow
    globals()["tensorflow"] = tensorflow
    return tensorflow # type: ignore
