import argparse
import enum
import sys
import threading
import time
from types import ModuleType
from typing import Callable, Optional

from ..utils.lazyloading import lazy_module

ArgumentParser = argparse.ArgumentParser|argparse._ArgumentGroup|argparse._MutuallyExclusiveGroup
Configurator = Callable[[argparse.ArgumentParser], None]
Job = Callable[[argparse.Namespace], None]

class EarlyStop(Exception):
    pass

class State(enum.Enum):
    Idle = enum.auto()
    Running = enum.auto()
    Stopping = enum.auto()
    Finished = enum.auto()

# The internal state
__state = State.Idle

# The modules being used
__modules = []

# The active job thread
__job_thread: threading.Thread|None = None

# The argument parser for the job
__argument_parser: argparse.ArgumentParser|None = None

# The job configuration
__config: argparse.Namespace|None = None

# Scripting Interface ------------------------------------------------------------------------------

def argument_parser():
    """
    Get the argument parser for the current job
    """
    return __argument_parser


def config() -> argparse.Namespace:
    """
    Returns the configuration of the current job.
    """
    if __config is None:
        raise Exception("No job is running.")
    return __config


def is_running() -> bool:
    """
    Returns whether the current job is running.
    """
    return __state == State.Running


def is_using(module: str|ModuleType) -> bool:
    """
    Check if the given module is being used.
    """
    for used_module in __modules:
        if isinstance(module, str):
            if used_module.__name__ == module:
                return True
        else:
            if used_module == module:
                return True
    return False


def use(module: ModuleType):
    """
    Use a scripting module for the current job.
    """
    assert module not in __modules
    __modules.append(module)
    __modules.sort(key=lambda m: m.NAME)


def init(argument_parser: Optional[argparse.ArgumentParser] = None):
    """
    Initialize the current job.
    """
    global __argument_parser, __config, __modules, __state
    if argument_parser is None:
        argument_parser = argparse.ArgumentParser()
    __argument_parser = argument_parser
    __state = State.Idle
    __modules = []
    __config = None
    assert __job_thread is None


def run(
    job: Job,
    argument_parser: argparse.ArgumentParser = argparse.ArgumentParser(),
    argv: Optional[list[str]] = None
):
    """
    Run the given job.
    """
    global __config, __state
    if __state != State.Idle:
        raise Exception("A job can only be run once.")
    __state = State.Running
    __define_arguments(argument_parser)
    __config = argument_parser.parse_args(argv)
    try:
        __init(__config)
        __start(__config)
        __run(job, __config)
    except KeyboardInterrupt:
        print("Keyboard interrupted. Stopping...")
    except EarlyStop:
        print("Stopping...")
    __state = State.Stopping
    __stop(__config)
    __finish(__config)
    __state = State.Finished


# Scripting Implementation -------------------------------------------------------------------------


def __define_arguments(parser: argparse.ArgumentParser):
    """
    Load the arguments for each module.
    """
    for module in __modules:
        if __state != State.Running:
            raise EarlyStop()
        module._define_arguments(parser)


def __init(config: argparse.Namespace):
    """
    Initialize each module.
    """
    for module in __modules:
        if __state != State.Running:
            raise EarlyStop()
        module._init(config)


def __start(config: argparse.Namespace):
    """
    Start each module.
    """
    for module in __modules:
        if __state != State.Running:
            raise EarlyStop()
        module._start(config)


def __run(job: Job, config: argparse.Namespace):
    """
    Run the given job.
    """
    global __job_thread
    __job_thread = threading.Thread(target=job, args=(config,), daemon=True)
    __job_thread.start()
    while __job_thread.is_alive() and __state == State.Running:
        time.sleep(0)


def __stop(config: argparse.Namespace):
    """
    Stop the job by waiting on the thread and stopping each module.
    """
    global __job_thread
    if __job_thread is not None:
        __job_thread.join()
        __job_thread = None
    for module in __modules:
        module._stop(config)


def __finish(config: argparse.Namespace):
    """
    Finish each module.
    """
    for module in __modules:
        module._finish(config)


# Module Imports -----------------------------------------------------------------------------------

@lazy_module
def __import(): # type: ignore
    from .module import rng
    return rng
rng = __import

@lazy_module
def __import(): # type: ignore
    from .module import train
    return train
train = __import

@lazy_module
def __import(): # type: ignore
    from .module import wandb
    return wandb
wandb = __import
