import argparse
from typing import Optional
import wandb
from ... import scripting as dls

# Module Interface ---------------------------------------------------------------------------------

def add_artifact_argument(
    parser: dls.ArgumentParser,
    name: str,
    required: bool = False,
    title: Optional[str] = None,
    description: Optional[str] = None
):
    """
    Add an artifact argument to the given parser.
    """
    title = title if title is not None else f"{name} artifact"
    group = parser.add_argument_group(title=title, description=description)
    group = group.add_mutually_exclusive_group(required=required)
    group.add_argument(f"--{name}-path")
    group.add_argument(f"--{name}-artifact")

def config_exclude_keys() -> set[str]|None:
    return __config_exclude_keys

def config_include_keys() -> set[str]|None:
    return __config_include_keys

def add_config_exclude_keys(keys: set|list[str]):
    __config_exclude_keys.update(keys)

def add_config_include_keys(keys: set|list[str]):
    __config_include_keys.update(keys)

def can_resume() -> bool:
    """
    Check if this job type is able to be resumed.
    """
    return __can_resume

def set_resumeable(can_resume: bool):
    """
    Set whether this job type is able to be resumed.
    """
    __can_resume = can_resume

def job_type() -> str|None:
    """
    Get the job type.
    """
    return __job_type

def set_job_type(job_type: str):
    """
    Set the job type.
    """
    global __job_type
    __job_type = job_type

# Module Configuration -----------------------------------------------------------------------------

NAME = "Weights & Biases"

__argument_parser: dls.ArgumentParser|None = None

__can_resume: bool = True
__config_exclude_keys: set[str] = set([
    "wandb_project",
    "wandb_name",
    "wandb_entity",
    "wandb_group",
    "wandb_tags",
    "wandb_notes",
    "wandb_dir",
    "wandb_save_code",
    "wandb_resume",
    "wandb_mode"
])
__config_include_keys: set[str] = set()
__job_type: str|None = None

def argument_parser() -> dls.ArgumentParser:
    """
    Get the current argument parser
    """
    assert __argument_parser is not None
    return __argument_parser


def _define_arguments(parser: dls.ArgumentParser):
    """
    Descriptions pulled from W&B documentation:
    https://docs.wandb.ai/ref/python/init
    """
    global __argument_parser
    group = __argument_parser = parser.add_argument_group(title=NAME, description="Configuration for Weights & Biases.")
    group.add_argument("--wandb-project", type=str, required=False, default=None, help="The name of the project where you're sending the new run. If the project is not specified, the run is put in an \"Uncategorized\" project.")
    group.add_argument("--wandb-name", type=str, required=False, default=None, help="A short display name for this run, which is how you'll identify this run in the UI. By default, we generate a random two-word name that lets you easily cross-reference runs from the table to charts. Keeping these run names short makes the chart legends and tables easier to read. If you're looking for a place to save your hyperparameters, we recommend saving those in config.")
    group.add_argument("--wandb-entity", type=str, required=False, default=None, help="An entity is a username or team name where you're sending runs. This entity must exist before you can send runs there, so make sure to create your account or team in the UI before starting to log runs. If you don't specify an entity, the run will be sent to your default entity, which is usually your username. Change your default entity in your settings under \"default location to create new projects\".")
    group.add_argument("--wandb-group", type=str, required=False, default=None, help="Specify a group to organize individual runs into a larger experiment. For example, you might be doing cross validation, or you might have multiple jobs that train and evaluate a model against different test sets. Group gives you a way to organize runs together into a larger whole, and you can toggle this on and off in the UI. For more details, see our guide to grouping runs.")
    group.add_argument("--wandb-tags", type=str, required=False, default=None, help="A list of strings, which will populate the list of tags on this run in the UI. Tags are useful for organizing runs together, or applying temporary labels like \"baseline\" or \"production\". It's easy to add and remove tags in the UI, or filter down to just runs with a specific tag.")
    group.add_argument("--wandb-notes", type=str, required=False, default=None, help="A longer description of the run, like a -m commit message in git. This helps you remember what you were doing when you ran this run.")
    group.add_argument("--wandb-dir", type=str, required=False, default=None, help="An absolute path to a directory where metadata will be stored. When you call download() on an artifact, this is the directory where downloaded files will be saved. By default, this is the ./wandb directory.")
    group.add_argument("--wandb-save_code", action="store_true", required=False, default=False, help="Turn this on to save the main script or notebook to W&B. This is valuable for improving experiment reproducibility and to diff code across experiments in the UI. By default this is off, but you can flip the default behavior to on in your settings page.")
    group.add_argument("--wandb-mode", type=str, required=False, choices=["online", "offline", "disabled"], help="The logging mode.")
    if __can_resume:
        group.add_argument("--wandb-resume", type=str, required=False, default=None, help="Resume a previous run given its ID.")

def _init(config: argparse.Namespace):
    pass

def _start(config: argparse.Namespace):
    resume = "never"
    run_id: str|None = None
    if __can_resume and config.wandb_resume is not None:
        resume = "must"
        run_id = config.wandb_resume
    print(__config_exclude_keys)
    wandb.init(
        id=run_id,
        job_type=__job_type,
        project=config.wandb_project,
        name=config.wandb_name,
        entity=config.wandb_entity,
        group=config.wandb_group,
        tags=config.wandb_tags,
        notes=config.wandb_notes,
        dir=config.wandb_dir,
        save_code=config.wandb_save_code,
        mode=config.wandb_mode,
        resume=resume,
        config=wandb.helper.parse_config(
            config,
            exclude=__config_exclude_keys,
            include=__config_include_keys)
    )

def _stop(config: argparse.Namespace):
    pass

def _finish(config: argparse.Namespace):
    pass
