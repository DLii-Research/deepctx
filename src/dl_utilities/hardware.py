import re
import subprocess
from typing import cast, TypedDict

# Type Definitions ---------------------------------------------------------------------------------

GpuMemoryInfo = TypedDict("GpuMemoryInfo", {"used": int, "free": int, "total": int})

# Interface Functions ------------------------------------------------------------------------------

def gpu_memory() -> tuple[GpuMemoryInfo, ...]:
    """
    Get the memory usage of all GPUs on the system.

    Implementation inspired by: https://stackoverflow.com/a/59571639
    """
    command = "nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv"
    gpu_info = subprocess.check_output(command.split()).decode('ascii').rstrip().split('\n')[1:]
    return cast(tuple[GpuMemoryInfo, ...], tuple(({
        k: v for k, v in zip(("used", "free", "total"), map(int, re.findall(r"(\d+)", gpu)))
    } for gpu in gpu_info)))
