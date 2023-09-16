from datetime import datetime
import os
import re
import subprocess
from typing import cast, TypedDict

class SlurmJobInfo(TypedDict):
    JobId: int
    JobName: str
    UserId: str
    GroupId: str
    MCS_label: str
    Priority: int
    Nice: int
    Account: str
    QOS: str
    JobState: str
    Reason: str
    Dependency: str
    Requeue: int
    Restarts: int
    BatchFlag: int
    Reboot: int
    ExitCode: str
    RunTime: str
    TimeLimit: str
    TimeMin: str
    SubmitTime: datetime
    EligibleTime: datetime
    AccrueTime: datetime
    StartTime: datetime
    EndTime: datetime
    Deadline: str
    SuspendTime: str
    SecsPreSuspend: int
    LastSchedEval: datetime
    Scheduler: str
    Partition: str
    Sid: str
    ReqNodeList: str
    ExcNodeList: str
    NodeList: str
    BatchHost: str
    NumNodes: int
    NumCPUs: int
    NumTasks: str
    Task: str
    T: str
    TRES: str
    cpu: int
    node: str
    billing: str
    Node: str
    C: str
    CoreSpec: str
    MinCPUsNode: int
    MinMemoryNode: int
    MinTmpDiskNode: int
    Features: str
    DelayBoot: str
    OverSubscribe: str
    Contiguous: int
    Licenses: str
    Network: str
    Command: str
    WorkDir: str
    StdErr: str
    StdIn: str
    StdOut: str
    Power: str

def is_job():
    return "SLURM_JOB_ID" in os.environ

def job_id() -> int:
    return int(os.environ["SLURM_JOB_ID"])

def job_info():
    def __get_info_str():
        try:
            return subprocess.check_output(f"scontrol show job {job_id()}").decode()
        except FileNotFoundError:
            pass
        try:
            return os.environ["SLURM_JOB_INFO"]
        except KeyError:
            pass
        raise Exception(
            "Unable to obtain SLURM job information. Please provide job info explicitly by" \
            + " in the environment variable SLURM_JOB_INFO. This can be done by invoking the" \
            + " following before executing your script:\n\n" \
            + "\texport SLURM_JOB_INFO=$(scontrol show job $SLURM_JOB_ID)\n")
    info = __get_info_str()
    vars_and_values = re.split(r"([a-zA-Z0-9_]+)=", info.strip().replace('\n', ' '))[1:]
    result = {
        k: v.rstrip() for k, v in zip(vars_and_values[::2], vars_and_values[1::2])
    }
    for k, v in result.items():
        if v.isnumeric():
            result[k] = int(v)
        elif re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", v):
            result[k] = datetime.strptime(v, "%Y-%m-%dT%H:%M:%S")
    return cast(SlurmJobInfo, result)
