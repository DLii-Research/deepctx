import argparse
import bisect
import enum
import sys
import threading
import time
from typing import Callable, cast, Optional, overload, TypeVar

ArgumentParser = argparse.ArgumentParser|argparse._ArgumentGroup|argparse._MutuallyExclusiveGroup
ContextModuleType = TypeVar("ContextModuleType", bound="ContextModule")
Job = Callable[["Context"], None]

class EarlyStop(Exception):
    pass

class State(enum.Enum):
    Idle = enum.auto()
    Running = enum.auto()
    Stopping = enum.auto()
    Finished = enum.auto()

class ContextModule:
    NAME: str

    def __init__(self, context: "Context"):
        self._context = context

    def _define_arguments(self):
        pass

    def _init(self):
        pass

    def _start(self):
        pass

    def _stop(self):
        pass

    def _finish(self):
        pass

    @property
    def context(self) -> "Context":
        """
        Get the context for this module.
        """
        return self._context

class Context:

    # A mapping of threads to contexts
    contexts: dict[threading.Thread, "Context"] = {}

    def __init__(
        self,
        job: Job,
        program_name: Optional[str] = None,
        description: Optional[str] = None,
        epilog: Optional[str] = None,
        argv: Optional[list[str]] = None
    ):
        self._job = job
        self._argv = argv
        self._config: argparse.Namespace|None = None
        self._modules: list[ContextModule] = []
        self._state = State.Idle
        self._thread: threading.Thread|None = None
        self._argument_parser: argparse.ArgumentParser = argparse.ArgumentParser(
            prog=program_name,
            description=description,
            epilog=epilog
        )

    def use_argument_parser(self, argument_parser: argparse.ArgumentParser) -> "Context":
        """
        Use the given argument parser for this context.
        """
        self.__argument_parser = argument_parser
        return self

    @overload
    def get(self, module: type[ContextModuleType]) -> ContextModuleType:
        ...
    @overload
    def get(self, module: Callable[[], type[ContextModuleType]]) -> ContextModuleType:
        ...
    def get(
        self,
        module: type[ContextModuleType]|Callable[[], type[ContextModuleType]]
    ) -> ContextModuleType:
        """
        Get the given module for the current context.
        """
        if isinstance(module, Callable):
            module = module()
        for used_module in self._modules:
            if isinstance(used_module, module):
                return cast(ContextModuleType, used_module)
        raise Exception(f"Module {module} is not being used.")

    @overload
    def is_using(self, module: type[ContextModuleType]) -> bool:
        ...
    @overload
    def is_using(self, module: Callable[[], type[ContextModuleType]]) -> bool:
        ...
    def is_using(self, module: type[ContextModuleType]|Callable[[], type[ContextModuleType]]) -> bool:
        """
        Check if the given module is being used in this context.
        """
        if isinstance(module, Callable):
            module = module()
        return any(isinstance(used_module, module) for used_module in self._modules)

    @overload
    def use(self, module: type[ContextModule]) -> ContextModuleType:
        ...
    @overload
    def use(self, module: Callable[[], type[ContextModuleType]]) -> ContextModuleType:
        ...
    def use(self, module: type[ContextModuleType]|Callable[[], type[ContextModuleType]]) -> ContextModuleType:
        """
        Use the given module in this context.
        """
        if isinstance(module, Callable):
            module = module()
        assert module not in self._modules
        instance = module(self)
        bisect.insort(self._modules, instance, key=lambda m: m.NAME)
        return instance

    def execute(self):
        execute(self)

    def _run(self):
        """
        Run the given job.
        """
        self.contexts[threading.current_thread()] = self
        if self._state == State.Running:
            for module in self._modules:
                module._init()
        if self._state == State.Running:
            for module in self._modules:
                module._start()
        if self._state == State.Running:
            self._job(self)
            self._state = State.Stopping
        for module in self._modules:
            module._stop()
        for module in self._modules:
            module._finish()
        self._state = State.Finished
        del self.contexts[threading.current_thread()]

    def _start(self):
        """
        Start the current context.
        """
        assert self._state == State.Idle, "Context is already running."
        self._state = State.Running
        for module in self._modules:
            module._define_arguments()
        self._config = self._argument_parser.parse_args(self._argv)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """
        Stop the current context
        """
        assert self._state == State.Running, "Context is not running."
        print("Stopping...")
        self._state = State.Stopping

    def wait_for_finish(self):
        """
        Finish the current context
        """
        assert self._state != State.Idle, "Context is not running."
        if self._thread is not None:
            self._thread.join()

    @property
    def argument_parser(self) -> argparse.ArgumentParser:
        """
        Get the argument parser for this context.
        """
        return self._argument_parser

    @property
    def config(self) -> argparse.Namespace:
        """
        Get the configuration for this context.
        """
        assert self._config is not None, "No configuration available."
        return self._config

    @property
    def is_running(self) -> bool:
        """
        Check if this context is still running.
        """
        if self._thread is None:
            return False
        return self._state == State.Running and self._thread.is_alive()

    @property
    def state(self) -> State:
        """
        Get the state of the current context.
        """
        return self._state


def context() -> Context:
    """
    Get the current context.
    """
    assert threading.current_thread() in Context.contexts, "No active context."
    return Context.contexts[threading.current_thread()]


def execute(*contexts: Context):
    """
    Execute the given contexts.
    """
    try:
        for context in contexts:
            context._start()
        while any(context.is_running for context in contexts):
            time.sleep(0)
    except KeyboardInterrupt:
        print("Keyboard interrupt")
        for context in contexts:
            context.stop()
    for context in contexts:
        context.wait_for_finish()
