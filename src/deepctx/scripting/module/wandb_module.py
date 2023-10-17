import abc
import argparse
import enum
import os
from pathlib import Path
import shutil
from typing import Callable, cast, Generic, Optional, TypedDict, TypeVar, TYPE_CHECKING
from .. import ArgumentParser
from ..context import Context, ContextModule
from ...lazy import wandb

if TYPE_CHECKING:
    from wandb.wandb_run import Run

T = TypeVar("T")
DoResult = TypeVar("DoResult")

class ArtifactArgumentInfo(TypedDict):
    """
    Information about an artifact argument.
    """
    name: str
    type: str|None
    aliases: list[str]|None
    use_as: str|None

class WandbRunDefaults(TypedDict):
    wandb_project: str|None
    wandb_name: str|None
    wandb_entity: str|None
    wandb_group: str|None
    wandb_tags: list[str]|None
    wandb_notes: str|None
    wandb_dir: str|None
    wandb_save_code: bool|None

class PersistentObject(abc.ABC, Generic[T]):
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
        self._state = PersistentObject.State.Idle
        self._to_save: list[Path] = []
        self.context.get(Wandb)._persistent_objects.append(self) # Register this factory with W&B

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
        self._state = PersistentObject.State.Idle
        return result

    def _create(self) -> T:
        return self._do(
            lambda: self.create(self.context.config),
            PersistentObject.State.Creating)

    def _load(self) -> T:
        return self._do(self.load, PersistentObject.State.Loading)

    def _save(self):
        if self._instance is None:
            return
        result = self._do(self.save, PersistentObject.State.Saving)
        for path in self._to_save:
            self.wandb.run.save(str(path))
        self._to_save.clear()
        return result

    def path(self, path: str|Path) -> Path:
        assert not Path(path).is_absolute(), "Absolute paths are not allowed in persistent object factories."
        path = Path("persistent_objects") / path
        abs_path = Path(self.wandb.run.dir) / path
        if self._state == PersistentObject.State.Loading:
            self.wandb.restore(path)
        elif self._state == PersistentObject.State.Saving:
            self._to_save.append(path)
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

class WandbApi(ContextModule):

    NAME = "Weights & Biases API"

    def __init__(self, context: Context):
        super().__init__(context)
        self._api: "wandb.Api|None" = None
        self._argument_parser: ArgumentParser|None = None
        self._config_artifact_arguments: dict[str, ArtifactArgumentInfo] = {}
        self._config_artifacts: dict[str, Path] = {}

    # Module Interface -----------------------------------------------------------------------------

    @property
    def api(self) -> "wandb.Api":
        """
        Get the W&B API.
        """
        if self._api is None:
            self._api = wandb.Api()
        return self._api

    # Module Configuration -------------------------------------------------------------------------

    @property
    def argument_parser(self) -> ArgumentParser:
        """
        Grab an argument parser for this module.
        """
        if self._argument_parser is None:
            self._argument_parser = self.context.argument_parser.add_argument_group(
                title=self.NAME,
                description="Configuration for the Weights & Biases module.")
        return self._argument_parser

    def add_artifact_argument(
        self,
        name: str,
        type: Optional[str] = None,
        aliases: Optional[list[str]] = None,
        use_as: Optional[str] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        required: bool = False,
        default: Optional[str] = None,
        parser: Optional[ArgumentParser] = None
    ):
        """
        Add an artifact argument to the given argument parser. This includes a mutually exclusive
        group with two options: `--{name}-artifact` and `--{name}-path`.

        If a W&B artifact is specified, it will automatically be downloaded.
        """
        key = name.replace("-", "_")
        assert key not in self._config_artifact_arguments, f"Artifact with key: `{name}` has already been added."
        parser = parser if parser is not None else self.context.argument_parser
        title = title if title is not None else name
        group = parser.add_argument_group(title=title, description=description)
        group = group.add_mutually_exclusive_group(required=required)
        group.add_argument(f"--{name}-artifact", type=str, default=default, help="The name of the Weights & Biases artifact to use.")
        group.add_argument(f"--{name}-path", type=str, help="The local path to the artifact data.")
        self._config_artifact_arguments[key] = ArtifactArgumentInfo(
            name=name,
            type=type,
            aliases=aliases,
            use_as=use_as)

    def artifact_argument_path(self, name: str) -> Path:
        """
        Get the path of a provided artifact argument. If a local path is specifie, the path is
        returned. If a W&B artifact is provided, it will automatically be downloaded.
        """
        key = name.replace("-", "_")
        assert key in self._config_artifact_arguments, f"Artifact with key: `{name}` has not been added."
        if name not in self._config_artifacts:
            config = self.context.config
            if getattr(config, f"{name}_artifact") is not None:
                artifact_info = self._config_artifact_arguments[key]
                artifact = self.use_artifact(
                    artifact_or_name=getattr(config, f"{key}_artifact"),
                    type=artifact_info["type"],
                    aliases=artifact_info["aliases"],
                    use_as=artifact_info["use_as"])
                self._config_artifacts[key] = Path(artifact.download())
            elif getattr(config, f"{name}_path") is not None:
                self._config_artifacts[key] = Path(getattr(config, f"{key}_path"))
            else:
                raise RuntimeError(f"No artifact or path provided for: `{name}`.")
        return self._config_artifacts[name]

    def use_artifact(
        self,
        artifact_or_name: "str|wandb.Artifact",
        type: Optional[str] = None,
        aliases: Optional[list[str]] = None,
        use_as: Optional[str] = None
    ) -> "wandb.Artifact":
        """
        Use the given artifact.
        """
        return self.api.artifact(artifact_or_name, type)


class Wandb(WandbApi):
    NAME = "Weights & Biases"

    Api = WandbApi
    PersistentObject = PersistentObject
    wandb = wandb

    def __init__(self, context: Context):
        super().__init__(context)
        self._run: Run|None = None
        self._api_only = False
        self._job_type: str|None = None
        self._can_resume: bool = False
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
        self._persistent_objects: list[PersistentObject] = []
        self._defaults: WandbRunDefaults = {
            "wandb_project": None,
            "wandb_name": None,
            "wandb_entity": None,
            "wandb_group": None,
            "wandb_tags": None,
            "wandb_notes": None,
            "wandb_dir": None,
            "wandb_save_code": None
        }

    # Module Interface -----------------------------------------------------------------------------

    @property
    def can_resume(self):
        """
        Check if this job is able to be resumed.
        """
        return self._can_resume

    @property
    def run(self) -> "Run": # implicit type to ensure lazy importing
        """
        Get the current run.
        """
        if self._api_only:
            raise RuntimeError("No Weights & Biases run instance available when using API only.")
        assert self._run is not None
        return self._run

    def log_artifact(
        self,
        artifact: "wandb.Artifact|Path|str",
        name: Optional[str] = None,
        type: Optional[str] = None,
        aliases: Optional[list[str]] = None
    ) -> "wandb.Artifact":
        """
        Log the given artifact.
        """
        return self.run.log_artifact(artifact, name, type, aliases)

    def restore(
        self,
        name: str|Path,
        run_path: Optional[str|Path] = None,
        replace: bool = False,
        root: Optional[str|Path] = None
    ) -> Path:
        """
        Restore (recursively) a the given directory from a previous run.
        """
        name = Path(name)
        run_path = Path(run_path if run_path is not None else self.run.path)
        run = self.api.run(str(run_path))
        found = False
        for f in filter(lambda f: str(f.name).startswith(str(name)), run.files()):
            self.run.restore(f.name, str(run_path), replace, root)
            found = True
        if not found:
            print("Warning: file(s) not found online. Searching previous local runs...")
            for run_dir in sorted(
                (Path(self.run.dir) / "../../").glob(f"*{self.run.id}"),
                reverse=True
            ):
                path = run_dir / "files" / name
                if not path.exists():
                    continue
                print(f"Copying '{path}' to '{self.run.dir / name}'")
                copy = shutil.copy if path.is_file() else shutil.copytree
                copy(path, self.run.dir / name)
                found = True
                break
        if not found:
            raise RuntimeError(f"File(s) not found: `{name}`.")
        return Path(self.run.dir) / name

    def use_artifact(
        self,
        artifact_or_name: "str|wandb.Artifact",
        type: Optional[str] = None,
        aliases: Optional[list[str]] = None,
        use_as: Optional[str] = None
    ) -> "wandb.Artifact":
        """
        Use the given artifact.
        """
        if self._api_only or self.run.disabled:
            return super().use_artifact(artifact_or_name, type, aliases, use_as)
        return self.run.use_artifact(artifact_or_name, type, aliases, use_as)

    # Module Configuration -------------------------------------------------------------------------

    def add_artifact_argument(
        self,
        name: str,
        type: Optional[str] = None,
        aliases: Optional[list[str]] = None,
        use_as: Optional[str] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        required: bool = False,
        default: Optional[str] = None,
        parser: Optional[ArgumentParser] = None
    ):
        """
        Add an artifact argument to the given argument parser. This includes a mutually exclusive
        group with two options: `--{name}-artifact` and `--{name}-path`.

        If a W&B artifact is specified, it will automatically be downloaded.
        """
        super().add_artifact_argument(
            name,
            type,
            aliases,
            use_as,
            title,
            description,
            required,
            default,
            parser)
        self.exclude_config_keys([f"{name}_artifact"])

    def api_only(self, api_only: bool = True) -> "Wandb":
        """
        Only use the W&B API without creating a run.
        """
        self._api_only = api_only
        return self

    def defaults(
        self,
        *,
        project: str|None = None,
        name: str|None = None,
        entity: str|None = None,
        group: str|None = None,
        tags: str|None = None,
        notes: str|None = None,
        dir: str|None = None,
        save_code: str|None = None
    ) -> "Wandb":
        """
        Set default parameters for the W&B run.
        """
        self._defaults["wandb_project"] = project
        self._defaults["wandb_name"] = name
        self._defaults["wandb_entity"] = entity
        self._defaults["wandb_group"] = group
        self._defaults["wandb_tags"] = tags
        self._defaults["wandb_notes"] = notes
        self._defaults["wandb_dir"] = dir
        self._defaults["wandb_save_code"] = save_code
        return self

    def exclude_config_keys(self, keys: set[str]|list[str]) -> "Wandb":
        """
        Add keys to exclude from the config.
        """
        self._config_exclude_keys.update(keys)
        return self

    def include_config_keys(self, keys: set[str]|list[str]) -> "Wandb":
        """
        Add keys to include in the config.
        """
        self._config_include_keys.update(keys)
        return self

    def job_type(self, job_type: str|None) -> "Wandb":
        """
        Set the job type.
        """
        self._job_type = job_type
        return self

    def resumeable(self, resumeable: bool = True) -> "Wandb":
        """
        Check if this job type is able to be resumed.
        """
        if self._api_only:
            raise RuntimeError("Cannot set Weights & Biases run to be resumeable when using API only.")
        self._can_resume = resumeable
        return self

    # Module Lifecycle -----------------------------------------------------------------------------

    def _define_arguments(self):
        if self._api_only:
            return
        group = self.argument_parser
        group.add_argument("--wandb-project", type=str, required=False, default=None, help="The name of the project where you're sending the new run. If the project is not specified, the run is put in an \"Uncategorized\" project.")
        group.add_argument("--wandb-name", type=str, required=False, default=None, help="A short display name for this run, which is how you'll identify this run in the UI. By default, we generate a random two-word name that lets you easily cross-reference runs from the table to charts. Keeping these run names short makes the chart legends and tables easier to read. If you're looking for a place to save your hyperparameters, we recommend saving those in config.")
        group.add_argument("--wandb-entity", type=str, required=False, default=None, help="An entity is a username or team name where you're sending runs. This entity must exist before you can send runs there, so make sure to create your account or team in the UI before starting to log runs. If you don't specify an entity, the run will be sent to your default entity, which is usually your username. Change your default entity in your settings under \"default location to create new projects\".")
        group.add_argument("--wandb-group", type=str, required=False, default=None, help="Specify a group to organize individual runs into a larger experiment. For example, you might be doing cross validation, or you might have multiple jobs that train and evaluate a model against different test sets. Group gives you a way to organize runs together into a larger whole, and you can toggle this on and off in the UI. For more details, see our guide to grouping runs.")
        group.add_argument("--wandb-tags", type=lambda x: x.split(','), required=False, default=None, help="A list of strings, which will populate the list of tags on this run in the UI. Tags are useful for organizing runs together, or applying temporary labels like \"baseline\" or \"production\". It's easy to add and remove tags in the UI, or filter down to just runs with a specific tag.")
        group.add_argument("--wandb-notes", type=str, required=False, default=None, help="A longer description of the run, like a -m commit message in git. This helps you remember what you were doing when you ran this run.")
        group.add_argument("--wandb-dir", type=str, required=False, default=None, help="An absolute path to a directory where metadata will be stored. When you call download() on an artifact, this is the directory where downloaded files will be saved. By default, this is the ./wandb directory.")
        group.add_argument("--wandb-save_code", action="store_true", required=False, default=None, help="Turn this on to save the main script or notebook to W&B. This is valuable for improving experiment reproducibility and to diff code across experiments in the UI. By default this is off, but you can flip the default behavior to on in your settings page.")
        group.add_argument("--wandb-mode", type=str, required=False, choices=["online", "offline", "disabled"], default="online", help="The logging mode.")
        if self.can_resume:
            group.add_argument("--wandb-resume", type=str, required=False, default=None, help="Resume a previous run given its ID.")

    def _start(self):
        if self._api_only:
            return
        self._create_wandb_run()

    def _create_wandb_run(self):
        """
        Create the W&B run instance.
        """
        assert not self._api_only
        config = self.context.config

        # Run resuming
        resume = "never"
        run_id: str|None = None
        if self.can_resume and config.wandb_resume is not None:
            resume = "must"
            run_id = config.wandb_resume

        # Handling default parameters
        def parameter(key: str, parse = lambda x: x):
            if getattr(config, key) is not None:
                return getattr(config, key)
            if key.upper() in os.environ:
                return parse(os.environ[key.upper()])
            return self._defaults[key]

        # Run creation
        self._run = cast("Run", wandb.init(
            id=run_id,
            job_type=self._job_type,
            project=parameter("wandb_project"),
            name=parameter("wandb_name"),
            entity=parameter("wandb_entity"),
            group=parameter("wandb_group"),
            tags=parameter("wandb_tags", lambda x: x.split(',')),
            notes=parameter("wandb_notes"),
            dir=parameter("wandb_dir"),
            save_code=getattr(config, "wandb_save_code", self._defaults["wandb_save_code"]),
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
        for obj in self._persistent_objects:
            obj._save()
        self._upload_persistent_objects()
        self._run.finish()

    def _upload_persistent_objects(self):
        if self._api_only:
            return
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
