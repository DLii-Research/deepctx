from typing import Optional, TypedDict
from .. import ArgumentParser
from ..context import Context, ContextModule
from ... import scripting as dcs

class TrainConfigDefaults(TypedDict, total=False):
    """
    The default values for the training module.
    """
    accumulation_steps: int|None
    epochs: int|float # float for inf
    batch_size: int
    val_batch_size: int
    steps_per_epoch: int|None
    val_steps_per_epoch: int|None
    workers: int

class Train(ContextModule):

    NAME = "Training & Evaluation"

    def __init__(self, context: Context):
        super().__init__(context)
        self._defaults: TrainConfigDefaults = {}
        self._use_gradient_accumulation = False
        self._use_multiprocessing = True
        self._use_optional_training = False
        self._use_steps = False
        self._train_argument_parser: ArgumentParser|None = None
        self._validation_argument_parser: ArgumentParser|None = None

    # Module Interface -----------------------------------------------------------------------------

    @property
    def batch_size(self) -> int:
        """
        Get the batch size adjusted for gradient accumulation.
        """
        if self._use_gradient_accumulation:
            return self.context.config.batch_size//self.context.config.accumulation_steps
        return self.context.config.batch_size

    @property
    def steps_per_epoch(self) -> int:
        """
        Get the steps per epoch adjusted for gradient accumulation.
        """
        assert self._use_steps
        if self._use_gradient_accumulation:
            return self.context.config.steps_per_epoch*self.context.config.accumulation_steps
        return self.context.config.steps_per_epoch

    def fit(self, model, x, y=None, validation_data=None, callbacks: Optional[list] = None, **kwargs):
        """
        A simple wrapper to fit that automatically pulls values provided in the configuration and
        links to Weights & Biases if available.
        """
        if self._use_optional_training and not self.context.config.train:
            return
        config = self.context.config
        callbacks = list(callbacks or [])
        if self.context.is_using(dcs.module.Tensorflow):
            callback = self.context.get(dcs.module.Tensorflow).context_stopping_callback()
            callbacks.append(callback)
        if self.context.is_using(dcs.module.Wandb):
            callbacks.append(self.context.get(dcs.module.Wandb).wandb.keras.WandbMetricsLogger())
        batch_size = config.batch_size
        if self._use_gradient_accumulation:
            batch_size = config.batch_size//self._use_gradient_accumulation
        steps_per_epoch = getattr(config, "steps_per_epoch", None)
        if steps_per_epoch is not None and self._use_gradient_accumulation:
            steps_per_epoch = steps_per_epoch*config.accumulation_steps
        return model.fit(
            x,
            y,
            callbacks=callbacks,
            batch_size=batch_size,
            validation_data=validation_data,
            epochs=config.epochs,
            initial_epoch=config.initial_epoch or 0,
            steps_per_epoch=steps_per_epoch,
            validation_steps=getattr(config, "val_steps_per_epoch", None),
            workers=getattr(config, "workers", None),
            use_multiprocessing=getattr(config, "workers", 0) > 0,
            **kwargs)

    # Module Configuration -------------------------------------------------------------------------

    @property
    def train_argument_parser(self) -> dcs.ArgumentParser:
        """
        Get the argument parser for this module.
        """
        if self._train_argument_parser is None:
            self._train_argument_parser = self.context.argument_parser.add_argument_group("Training Settings")
        return self._train_argument_parser

    @property
    def validation_argument_parser(self) -> dcs.ArgumentParser:
        """
        Get the argument parser for this module.
        """
        if self._validation_argument_parser is None:
            self._validation_argument_parser = self.context.argument_parser.add_argument_group("Validation Settings")
        return self._validation_argument_parser

    def defaults(
        self,
        *,
        accumulation_steps: Optional[int] = ...,
        epochs: Optional[int|float] = ...,
        batch_size: Optional[int] = ...,
        val_batch_size: Optional[int] = ...,
        steps_per_epoch: Optional[int|None] = ...,
        val_steps_per_epoch: Optional[int|None] = ...,
        workers: Optional[int] = ...
    ) -> "Train":
        defaults = {
            "accumulation_steps": accumulation_steps,
            "epochs": epochs,
            "batch_size": batch_size,
            "val_batch_size": val_batch_size,
            "steps_per_epoch": steps_per_epoch,
            "val_steps_per_epoch": val_steps_per_epoch,
            "workers": workers
        }
        for key, value in defaults.items():
            if value is not Ellipsis:
                self._defaults[key] = value
        return self

    def multiprocessing(self, multiprocessing: bool = True) -> "Train":
        """
        Set whether or not to enable multiprocessing for data generation.
        """
        self._use_multiprocessing = multiprocessing
        return self

    def optional_training(self, optional_training: bool = True) -> "Train":
        """
        Set whether to enable optional training.
        """
        self._use_optional_training = optional_training
        return self

    def use_gradient_accumulation(self, gradient_accumulation: bool = True) -> "Train":
        """
        Set whether to enable gradient accumulation.
        """
        self._use_gradient_accumulation = gradient_accumulation
        return self

    def use_steps(self, use_steps: bool = True) -> "Train":
        """
        Use training steps instead of traditional epochs.
        """
        self._use_steps = use_steps
        return self

    # Module Lifecycle -----------------------------------------------------------------------------

    def _define_arguments(self):
        """
        Descriptions pulled from W&B documentation:
        https://docs.wandb.ai/ref/python/init
        """
        # Training Settings
        train = self.train_argument_parser
        train.add_argument("--epochs", type=lambda x: None if x == x.lower() == "inf" else int(x), default=self._defaults.get("epochs", 1), help="The number of epochs to train. Use --epochs inf for indefinite training.")
        train.add_argument("--initial-epoch", type=int, default=self._defaults.get("initial_epoch", None), help="The initial training epoch to start at.")
        train.add_argument("--batch-size", type=int, default=self._defaults.get("batch_size", 32), help="The training batch size to use.")
        if self._use_steps:
            train.add_argument("--steps-per-epoch", type=int, default=self._defaults.get("steps_per_epoch", None), required=("steps_per_epoch" not in self._defaults), help="The number of steps per epoch to use for training.")
        if self._use_gradient_accumulation:
            train.add_argument("--accumulation-steps", type=int, default=self._defaults.get("accumulation_steps", 1), help="Accumulate gradients over this many steps.")
        if self._use_optional_training:
            train.add_argument("--train", action="store_true", help="Train the model.")
        if self._use_multiprocessing:
            train.add_argument("--workers", type=int, default=self._defaults.get("workers", 0), help="The number of multi-processing workers to use for data generation.")

        #Validation Settings
        val = self.validation_argument_parser
        val.add_argument("--val-batch-size", type=int, default=self._defaults.get("val_batch_size", None), help="The validation batch size to use.")
        if self._use_steps:
            val.add_argument("--val-steps-per-epoch", type=int, default=self._defaults.get("val_steps_per_epoch", None), required=("val_steps_per_epoch" not in self._defaults), help="The number of steps per epoch to use for validation.")

    def _init(self):
        config = self.context.config
        if self._use_gradient_accumulation and config.batch_size % config.accumulation_steps != 0:
            raise ValueError("Batch size must be divisible by accumulation steps.")
        if config.epochs == float("inf") or config.epochs is None:
            config.epochs = 2**32 - 1
        if config.val_batch_size is None:
            config.val_batch_size = config.batch_size
        if self._use_steps and config.steps_per_epoch is not None and config.val_steps_per_epoch is None:
            config.val_steps_per_epoch = config.steps_per_epoch
        if self.context.is_using(dcs.module.Wandb):
            wandb = self.context.get(dcs.module.Wandb)
            wandb.exclude_config_keys([
                "epochs",
                "initial_epoch"
            ])

    def _ready(self):
        config = self.context.config
        if self.context.is_using(dcs.module.Wandb):
            wandb = self.context.get(dcs.module.Wandb)
            if config.initial_epoch is None:
                config.initial_epoch = wandb.run.step
