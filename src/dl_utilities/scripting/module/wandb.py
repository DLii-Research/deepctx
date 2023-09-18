import abc
from pathlib import Path
from typing import cast
import wandb
from wandb.wandb_run import Run
from typing import Generic, TypeVar
from .. import ArgumentParser
from ..context import Context, ContextModule

T = TypeVar("T")

class PersistentObjectFactory(abc.ABC, Generic[T]):
    """
    A factory for creating persistent objects that can be saved and loaded via W&B.
    """
    def __init__(self, context: Context, wandb: "Wandb"):
        self._context = context
        self._wandb = wandb
        self._object: T|None = None

    @property
    def context(self) -> Context:
        """
        Get the context.
        """
        return self._context

    @property
    def wandb(self) -> "Wandb":
        """
        Get the W&B module.
        """
        return self._wandb

    @abc.abstractmethod
    def save(self):
        pass

    @abc.abstractmethod
    def load(self) -> T:
        pass

    @abc.abstractmethod
    def create(self) -> T:
        pass

    def get(self) -> T:
        """
        Get the object.
        """
        if self._object is None:
            self._object = self.load()
            if self._object is None:
                self._object = self.create()
        return self._object


class Wandb(ContextModule):
    NAME = "Weights & Biases"

    PersistentObjectFactory = PersistentObjectFactory

    def __init__(self, context: Context):
        super().__init__(context)
        self._run: Run|None = None
        self._job_type: str|None = None
        self._can_resume: bool = True
        self._config_exclude_keys: set[str] = set([
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
        self._config_include_keys: set[str] = set()
        self._argument_parser = self.context.argument_parser.add_argument_group(
            title=self.NAME,
            description="Configuration for the Weights & Biases module.")

    @property
    def argument_parser(self) -> ArgumentParser:
        """
        Get the argument parser for this module.
        """
        assert self._argument_parser is not None
        return self._argument_parser

    @property
    def run(self) -> Run:
        """
        Get the current run.
        """
        assert self._run is not None
        return self._run

    def job_type(self, job_type: str|None) -> "Wandb":
        """
        Set the job type.
        """
        self._job_type = job_type
        return self

    @property
    def can_resume(self):
        return self._can_resume

    def resumeable(self, resumeable: bool = True) -> "Wandb":
        """
        Check if this job type is able to be resumed.
        """
        self._is_resumeable = resumeable
        return self

    def exclude_config_keys(self, keys: set|list[str]) -> "Wandb":
        """
        Add keys to exclude from the config.
        """
        self._config_exclude_keys.update(keys)
        return self

    def include_config_keys(self, keys: set|list[str]) -> "Wandb":
        """
        Add keys to include in the config.
        """
        self._config_include_keys.update(keys)
        return self

    def _define_arguments(self):
        group = self.argument_parser
        group.add_argument("--wandb-project", type=str, required=False, default=None, help="The name of the project where you're sending the new run. If the project is not specified, the run is put in an \"Uncategorized\" project.")
        group.add_argument("--wandb-name", type=str, required=False, default=None, help="A short display name for this run, which is how you'll identify this run in the UI. By default, we generate a random two-word name that lets you easily cross-reference runs from the table to charts. Keeping these run names short makes the chart legends and tables easier to read. If you're looking for a place to save your hyperparameters, we recommend saving those in config.")
        group.add_argument("--wandb-entity", type=str, required=False, default=None, help="An entity is a username or team name where you're sending runs. This entity must exist before you can send runs there, so make sure to create your account or team in the UI before starting to log runs. If you don't specify an entity, the run will be sent to your default entity, which is usually your username. Change your default entity in your settings under \"default location to create new projects\".")
        group.add_argument("--wandb-group", type=str, required=False, default=None, help="Specify a group to organize individual runs into a larger experiment. For example, you might be doing cross validation, or you might have multiple jobs that train and evaluate a model against different test sets. Group gives you a way to organize runs together into a larger whole, and you can toggle this on and off in the UI. For more details, see our guide to grouping runs.")
        group.add_argument("--wandb-tags", type=lambda x: x.split(','), required=False, default=None, help="A list of strings, which will populate the list of tags on this run in the UI. Tags are useful for organizing runs together, or applying temporary labels like \"baseline\" or \"production\". It's easy to add and remove tags in the UI, or filter down to just runs with a specific tag.")
        group.add_argument("--wandb-notes", type=str, required=False, default=None, help="A longer description of the run, like a -m commit message in git. This helps you remember what you were doing when you ran this run.")
        group.add_argument("--wandb-dir", type=str, required=False, default=None, help="An absolute path to a directory where metadata will be stored. When you call download() on an artifact, this is the directory where downloaded files will be saved. By default, this is the ./wandb directory.")
        group.add_argument("--wandb-save_code", action="store_true", required=False, default=False, help="Turn this on to save the main script or notebook to W&B. This is valuable for improving experiment reproducibility and to diff code across experiments in the UI. By default this is off, but you can flip the default behavior to on in your settings page.")
        group.add_argument("--wandb-mode", type=str, required=False, choices=["online", "offline", "disabled"], default="online", help="The logging mode.")
        if self.can_resume:
            group.add_argument("--wandb-resume", type=str, required=False, default=None, help="Resume a previous run given its ID.")

    def _init(self):
        pass

    def _start(self):
        config = self.context.config
        resume = "never"
        run_id: str|None = None
        if self.can_resume and config.wandb_resume is not None:
            resume = "must"
            run_id = config.wandb_resume
        self._run = cast(Run, wandb.init(
            id=run_id,
            job_type=self._job_type,
            project=config.wandb_project,
            name=config.wandb_name,
            entity=config.wandb_entity,
            group=config.wandb_group,
            tags=config.wandb_tags,
            notes=config.wandb_notes,
            dir=config.wandb_dir,
            save_code=config.wandb_save_code,
            mode=config.wandb_mode,
            reinit=True,
            resume=resume,
            config=wandb.helper.parse_config(
                config,
                exclude=self._config_exclude_keys,
                include=self._config_include_keys)
        ))

    def _stop(self):
        if self._run is None:
            return
        self._run.finish()

context_module = Wandb
