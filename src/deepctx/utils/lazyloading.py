"""
This module provides a simple lazy-loading mechanism for various libraries.
"""
from typing import Callable, cast, Generic, Set, TypeVar

_lazy_attribute_blacklist: Set[str] = set([
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

T = TypeVar("T")

class LazyWrapper(Generic[T]):
    """
    Lazily load/create an object.
    """
    def __init__(self, name: str, factory: Callable[[], T]):
        self.__name = name
        self.__factory = factory
        self.__wrapped_object = None

    @property
    def __is_loaded__(self):
        return self.__wrapped_object is not None

    def __load__(self) -> T:
        if self.__wrapped_object is None:
            self.__wrapped_object = self.__factory()
        return self.__wrapped_object

    @property
    def __wrapped_object__(self):
        return self.__load__()

    def __call__(self, *args, **kwargs):
        obj = self.__wrapped_object__
        return obj(*args, **kwargs) # type: ignore

    def __getattr__(self, attr):
        if self.__wrapped_object is not None and attr in _lazy_attribute_blacklist:
            raise AttributeError(f"Attribute {attr} is not available.")
        return getattr(self.__wrapped_object__, attr)

    def __repr__(self):
        return f"LazyObject({self.__name})"


def lazy_wrapper(factory: Callable[[], T]) -> T:
    """
    A decorator to lazily evaluate a function while providing all available type information.

    In most situations, the import function must be defined in the following format:

    ```py
    @lazy_wrapper
    def name():
        import name
        return name
    ```

    When dealing with local imports, it may be necessary to delete the globals.

    ```py
    @lazy_wrapper
    def name():
        del globals()["name"]
        from . import name
        return name
    ```
    """
    return cast(T, LazyWrapper(factory.__name__, factory))
