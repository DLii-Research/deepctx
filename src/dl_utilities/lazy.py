from .utils.lazyloading import lazy_module

@lazy_module
def __import():
    del globals()["tensorflow"]
    import tensorflow
    return tensorflow
tensorflow = __import
