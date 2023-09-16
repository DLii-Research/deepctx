import argparse
from ... import scripting as dls

# Module Interface ---------------------------------------------------------------------------------

# Module Configuration -----------------------------------------------------------------------------

NAME = "Training Hyperparameters"

def argument_parser() -> dls.ArgumentParser:
    """
    Get the current argument parser
    """
    assert __argument_parser is not None
    return __argument_parser

def _define_arguments(parser: dls.ArgumentParser):
    """
    Descriptions pulled from W&B documentation:
    https://docs.wandb.ai/ref/python/init
    """
    group = parser.add_argument_group(title=NAME, description="Configuration for training.")
    group.add_argument("--epochs", type=str, required=False, default=1, help="The number of epochs to train for.")
    group.add_argument("--batch-size", type=str, required=False, default=32, help="The training batch size to use.")


def _init(config: argparse.Namespace):
    if dls.is_using("dl_utilities.scripting.module.wandb"):
        from . import wandb
        wandb.add_config_exclude_keys(["epochs"])

def _start(config: argparse.Namespace):
    pass

def _stop(config: argparse.Namespace):
    pass

def _finish(config: argparse.Namespace):
    pass
