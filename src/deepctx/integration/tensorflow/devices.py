import tensorflow as tf
from typing import cast, Iterable, List, Optional, TypedDict, Union

from ... import hardware

# Type Definitions ---------------------------------------------------------------------------------

class TfGpuInfo(hardware.GpuInfo):
    device: tf.config.PhysicalDevice

# Interface Functions ------------------------------------------------------------------------------

def best_gpus(
    gpus: Optional[List[TfGpuInfo]] = None,
    count: int = 1
) -> List[TfGpuInfo]:
    """
    Select the given number of GPUs. The selected devices are prioritized by their available memory.
    """
    if gpus is None:
        gpus = gpu_list()
    best_gpus = hardware.best_gpus(cast(List[hardware.GpuInfo], gpus), count=count)
    return cast(List[TfGpuInfo], best_gpus)


def cpu_list() -> List[tf.config.PhysicalDevice]:
    """
    Get the list of visible CPU devices.
    """
    return tf.config.list_physical_devices("CPU")


def gpu_list() -> List[TfGpuInfo]:
    """
    Get the list of visible GPU devices.
    """
    gpus = cast(List[TfGpuInfo], hardware.gpus())
    devices = tf.config.list_physical_devices("GPU")
    assert len(gpus) == len(devices), "GPU list length mismatch"
    for gpu, device in zip(gpus, devices):
        gpu["device"] = device
    return gpus


def use(
    *,
    cpus: Optional[Union[int, Iterable[tf.config.PhysicalDevice], Iterable[int], None]] = ...,
    gpus: Optional[Union[int, Iterable[TfGpuInfo], Iterable[int], None]] = ...,
    use_dynamic_memory: bool = True
) -> List[tf.config.PhysicalDevice]:
    """
    Select the specified devices.

    cpus: If given a number, the first n CPUs will be used. If given a list of devices/indices,
          those devices will be used.
    gpus: If given a number, the best n GPUs will be used. If given a list of devices/indices,
          those devices will be used.
    use_dynamic_memory: Use dynamic memory allocation.
    """
    if cpus is not Ellipsis:
        if cpus is None:
            cpus = []
        elif isinstance(cpus, int):
            cpus = cpu_list()[:cpus]
        elif len(cpus) > 0 and isinstance(cpus[0], int): # type: ignore
            cpus = [cpu_list()[i] for i in cpus] # type: ignore
        tf.config.set_visible_devices(cpus, "CPU")
    if gpus is not Ellipsis:
        if gpus is None:
            gpus = []
        elif isinstance(gpus, int):
            gpus = best_gpus(count=gpus)
        elif len(gpus) > 0 and isinstance(gpus[0], int): # type: ignore
            gpus = [gpu_list()[i] for i in gpus] # type: ignore
        tf.config.set_visible_devices([gpu["device"] for gpu in cast(List[TfGpuInfo], gpus)], "GPU")
        for info in cast(List[TfGpuInfo], gpus):
            tf.config.experimental.set_memory_growth(info["device"], use_dynamic_memory)
    return tf.config.get_visible_devices()
