from .utils.lazyloading import lazy_wrapper

@lazy_wrapper
def tensorflow():
    del globals()["tensorflow"]
    import tensorflow
    return tensorflow
