from .utils.lazyloading import lazy_wrapper

@lazy_wrapper
def tensorflow():
    del globals()["tensorflow"]
    import tensorflow
    globals()["tensorflow"] = tensorflow
    return tensorflow


@lazy_wrapper
def wandb():
    del globals()["wandb"]
    import wandb
    globals()["wandb"] = wandb
    return wandb
