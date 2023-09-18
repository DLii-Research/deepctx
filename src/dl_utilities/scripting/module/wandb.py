import abc
import argparse
import enum
from pathlib import Path
from typing import cast
import wandb
from wandb.wandb_run import Run
from typing import Callable, Generic, Optional, TypeVar
from .. import ArgumentParser
from ..context import Context, ContextModule

T = TypeVar("T")
DoResult = TypeVar("DoResult")

class PersistentObjectFactory(abc.ABC, Generic[T]):
    """
    A factory for creating persistent objects that can be saved and loaded via W&B.
    """
    class State(enum.Enum):
        Creating = enum.auto()
        Idle = enum.auto()
        Loading = enum.auto()
        Saving = enum.auto()

    def __init__(self, context: Optional[Context] = None):
        if context is None:
            context = Context.current()
        self._context = context
        self._instance: T = None # type: ignore
        self._state = PersistentObjectFactory.State.Idle
        self.context.get(Wandb)._factories.append(self) # Register this factory with W&B

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
        return self.context.get(Wandb)

    @property
    def instance(self) -> T:
        """
        Get the current object's instance.
        """
        if self._instance is None:
            if self.wandb.run.resumed:
                self._instance = self._load()
            else:
                self._instance = self._create()
        return self._instance

    def _do(self, operation: Callable[[], DoResult], state: State) -> DoResult:
        self._state = state
        result = operation()
        self._state = PersistentObjectFactory.State.Idle
        return result

    def _create(self) -> T:
        return self._do(
            lambda: self.create(self.context.config),
            PersistentObjectFactory.State.Creating)

    def _load(self) -> T:
        return self._do(self.load, PersistentObjectFactory.State.Loading)

    def _save(self):
        if self._instance is None:
            return
        return self._do(self.save, PersistentObjectFactory.State.Saving)

    def path(self, path: str|Path) -> Path:
        assert not Path(path).is_absolute(), "Absolute paths are not allowed in persistent object factories."
        path = Path("persistent_objects") / path
        abs_path = Path(self.wandb.run.dir) / path
        if self._state == PersistentObjectFactory.State.Loading:
            self.wandb.restore(path, recursive=True)
        return abs_path

    @abc.abstractmethod
    def create(self, config: argparse.Namespace) -> T:
        raise NotImplementedError()

    @abc.abstractmethod
    def load(self) -> T:
        raise NotImplementedError()

    @abc.abstractmethod
    def save(self):
        raise NotImplementedError()


class Wandb(ContextModule):
    NAME = "Weights & Biases"

    PersistentObjectFactory = PersistentObjectFactory

    def __init__(self, context: Context):
        super().__init__(context)
        self._api: wandb.Api|None = None
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
        self._factories: list[PersistentObjectFactory] = []

    @property
    def api(self) -> wandb.Api:
        """
        Get the W&B API.
        """
        if self._api is None:
            self._api = wandb.Api()
        return self._api

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

    def restore(
        self,
        name: str|Path,
        run_path: Optional[str|Path] = None,
        replace: bool = False,
        root: Optional[str|Path] = None,
        recursive: bool = False
    ) -> Path:
        """
        Restore (recursively) a the given directory from a previous run.
        """
        name = Path(name)
        run_path = Path(run_path if run_path is not None else self.run.path)
        run = self.api.run(str(run_path))
        if recursive:
            for f in filter(lambda f: str(f.name).startswith(str(name)), run.files()):
                self.run.restore(f.name, str(run_path), replace, root)
        else:
            self.run.restore(str(name), str(run_path), replace, root)
        return Path(self.run.dir) / name

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
        for factory in self._factories:
            factory._save()
        persistent_objects = Path(self.run.dir) / "persistent_objects"
        if persistent_objects.exists() and persistent_objects.is_dir():
            paths = [persistent_objects]
            while len(paths) > 0:
                path = paths.pop()
                for child in path.iterdir():
                    if child.is_dir():
                        paths.append(child)
                    else:
                        self.run.save(str(child), base_path=str(child))
        self._run.finish()

context_module = Wandb
