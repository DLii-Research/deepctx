import os
import subprocess
from typing import List, Optional, TypedDict

# Type Definitions ---------------------------------------------------------------------------------

class GpuInfo(TypedDict):
    uuid: str
    num_processes: int
    memory_used: int
    memory_free: int
    memory_total: int

# Interface Functions ------------------------------------------------------------------------------

def best_gpus(
    gpu_list: Optional[List[GpuInfo]] = None,
    count: Optional[int] = None
) -> List[GpuInfo]:
    """
    Select the given number of GPUs. The selected devices are prioritized by the number of processes
    currently running and the memory usage.
    """
    if gpu_list is None:
        gpu_list = gpus()
    if count is None:
        count = len(gpu_list)
    return sorted(gpu_list, key=lambda gpu: (gpu["num_processes"], -gpu["memory_free"]))[:count]

def gpus(ignore_cuda_visibility: bool = False) -> List[GpuInfo]:
    gpus: dict[str, GpuInfo] = {}
    # Memory usage info
    command = "nvidia-smi --query-gpu=gpu_uuid,memory.used,memory.free,memory.total --format=csv"
    memory_info = subprocess.check_output(command.split()).decode('ascii').rstrip().split('\n')[1:]
    for info in memory_info:
        uuid, memory_used, memory_free, memory_total = info.split(', ')
        gpus[uuid] = {
            "uuid": uuid,
            "num_processes": 0,
            "memory_used": int(memory_used.split(' ')[0]),
            "memory_free": int(memory_free.split(' ')[0]),
            "memory_total": int(memory_total.split(' ')[0]),
        }
    # Process info
    command = "nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv "
    process_info = subprocess.check_output(command.split()).decode('ascii').rstrip().split('\n')[1:]
    for info in process_info:
        uuid, _ = info.split(', ')
        if uuid not in gpus:
            continue
        gpus[uuid]["num_processes"] += 1
    gpu_list = list(gpus.values())
    if not ignore_cuda_visibility:
        visible_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        if visible_gpus is not None:
            indices = tuple(map(int, visible_gpus.split(',')))
            assert len(indices) == len(set(indices))
            gpu_list = [gpu_list[i] for i in indices]
    return gpu_list
