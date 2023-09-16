import argparse
from typing import Optional
from ... import scripting as dls

# Module Interface ---------------------------------------------------------------------------------

# Module Configuration -----------------------------------------------------------------------------

NAME = "Random Number Generator"

def _define_arguments(parser: dls.ArgumentParser):
    group = parser.add_argument_group(title=NAME, description="Configuration for the random number generator.")
    group.add_argument("--seed", type=int, default=None)

def _init(config: argparse.Namespace):
    pass

def _start(config: argparse.Namespace):
    pass

def _stop(config: argparse.Namespace):
    pass

def _finish(config: argparse.Namespace):
    pass
