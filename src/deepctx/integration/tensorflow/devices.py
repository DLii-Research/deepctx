import tensorflow as tf
from typing import Iterable, Optional, TypedDict

from ...hardware import gpu_memory

# Type Definitions ---------------------------------------------------------------------------------

GpuMemoryInfo = TypedDict("GpuMemoryInfo", {"used": int, "free": int, "total": int})

# Interface Functions ------------------------------------------------------------------------------

def best_gpus(
    gpus: Optional[list[tf.config.PhysicalDevice]] = None,
    count: int = 1
) -> list[tf.config.PhysicalDevice]:
    """
    Select the given number of GPUs. The selected devices are prioritized by their available memory.
    """
    memory_utilization = {gpu: memory["free"] for gpu, memory in zip(gpu_list(), gpu_memory())}
    if gpus is None:
        gpus = gpu_list()
    # Sort gpus list by memory utilization
    gpus.sort(key=lambda gpu: memory_utilization[gpu], reverse=True)
    return gpus[:count]


def cpu_list() -> list[tf.config.PhysicalDevice]:
    """
    Get the list of visible CPU devices.
    """
    return tf.config.list_physical_devices("CPU")


def gpu_list() -> list[tf.config.PhysicalDevice]:
    """
    Get the list of visible GPU devices.
    """
    return tf.config.list_physical_devices("GPU")


def use(
    *,
    cpus: Optional[int|Iterable[tf.config.PhysicalDevice]|Iterable[int]|None] = ...,
    gpus: Optional[int|Iterable[tf.config.PhysicalDevice]|Iterable[int]|None] = ...,
    use_dynamic_memory: bool = True
) -> list[tf.config.PhysicalDevice]:
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
        tf.config.set_visible_devices(gpus, "GPU")
        for device in gpus:
            tf.config.experimental.set_memory_growth(device, use_dynamic_memory)
    return tf.config.get_visible_devices()
