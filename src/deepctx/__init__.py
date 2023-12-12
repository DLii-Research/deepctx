from .utils.lazyloading import lazy_wrapper

__version__ = "0.0.21"

from . import hardware
from . import integration

# Integration --------------------------------------------------------------------------------------

@lazy_wrapper
def tf():
    from .integration import tensorflow
    return tensorflow
