from .utils.lazyloading import lazy_module

__version__ = "0.0.1"

from . import hardware
from . import integration

# Integration --------------------------------------------------------------------------------------

# Tensorflow
@lazy_module
def __import(): from .integration import tensorflow; return tensorflow
tf = __import
