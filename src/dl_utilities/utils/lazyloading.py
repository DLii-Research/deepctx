"""
This module provides a simple lazy-loading mechanism for various libraries.
"""
import inspect
import re
from typing import Callable, cast, Generic, TypeVar
from types import ModuleType

_lazy_attribute_blacklist: set[str] = set([
    # IPython
    "_ipython_canary_method_should_not_exist_",
    "_ipython_display_",
    "_repr_mimebundle_",
    "_repr_html_",
    "_repr_markdown_",
    "_repr_svg_",
    "_repr_png_",
    "_repr_pdf_",
    "_repr_jpeg_",
    "_repr_latex_",
    "_repr_json_",
    "_repr_javascript_"
])

Module = TypeVar("Module")

class LazyModule(Generic[Module]):
    def __init__(self, name: str, importer: Callable[[], Module]):
        self.__name = name
        self.__importer = importer
        self.__module = None

    @property
    def is_loaded(self):
        return self.__module is not None

    def __getattr__(self, attr):
        if self.__module is None:
            if attr in _lazy_attribute_blacklist:
                raise AttributeError(f"Attribute {attr} is not available.")
            if self.__name in globals():
                del globals()[self.__name]
            self.__module = self.__importer()
        return getattr(self.__module, attr)

    def __repr__(self):
        return f"LazyModule({self.__name})"


def lazy_module(importer: Callable[[], Module]) -> Module:
    """
    A decorator to lazily-import a module while providing all available type information.

    In most situations, the import function must be defined in the following format:

    '''py
    @lazymodule
    def name():
        import name
        return name
    ```
    """
    return lazy_import(importer)


def lazy_import(importer: Callable[[], Module]) -> Module:
    """
    Lazily-import a module while providing all available type information.

    In most situations, the import function must be defined in the following format:

    ```py
    def __import():
        import name
        return name
    name = lazy_import(__import)
    ```
    """
    name = re.findall(r"return ([^\s\n]*)", inspect.getsource(importer))[0]
    return cast(Module, LazyModule(cast(str, name), importer))
