from typing import Optional, TypedDict
from .. import ArgumentParser
from ..context import Context, ContextModule
from ... import scripting as dls

class TrainConfigDefaults(TypedDict, total=False):
    """
    The default values for the training module.
    """
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
        self._use_multiprocessing = True
        self._use_optional_training = False
        self._use_steps = False
        self._train_argument_parser: ArgumentParser|None = None
        self._validation_argument_parser: ArgumentParser|None = None

    # Module Interface -----------------------------------------------------------------------------

    def fit(self, model, x, y=None, validation_data=None, callbacks: list=None):
        """
        A simple wrapper to fit that automatically pulls values provided in the configuration and
        links to Weights & Biases if available.
        """
        if self._use_optional_training and not self.context.config.train:
            return
        config = self.context.config
        callbacks = callbacks or []
        if self.context.is_using(dls.module.Tensorflow):
            callback = self.context.get(dls.module.Tensorflow).ContextStoppingCallback(self.context)
            callbacks.append(callback)
        return model.fit(
            x,
            y,
            callbacks=callbacks,
            batch_size=config.batch_size,
            validation_data=validation_data,
            epochs=config.epochs,
            initial_epoch=config.initial_epoch or 0,
            steps_per_epoch=getattr(config, "steps_per_epoch", None),
            validation_steps=getattr(config, "val_steps_per_epoch", None),
            workers=getattr(config, "workers", None),
            use_multiprocessing=getattr(config, "workers", 0) > 0)

    # Module Configuration -------------------------------------------------------------------------

    @property
    def train_argument_parser(self) -> dls.ArgumentParser:
        """
        Get the argument parser for this module.
        """
        if self._train_argument_parser is None:
            self._train_argument_parser = self.context.argument_parser.add_argument_group("Training Settings")
        return self._train_argument_parser

    @property
    def validation_argument_parser(self) -> dls.ArgumentParser:
        """
        Get the argument parser for this module.
        """
        if self._validation_argument_parser is None:
            self._validation_argument_parser = self.context.argument_parser.add_argument_group("Validation Settings")
        return self._validation_argument_parser

    def defaults(
        self,
        *,
        epochs: Optional[int|float] = ...,
        batch_size: Optional[int] = ...,
        val_batch_size: Optional[int] = ...,
        steps_per_epoch: Optional[int|None] = ...,
        val_steps_per_epoch: Optional[int|None] = ...,
        workers: Optional[int] = ...
    ) -> "Train":
        defaults = {
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
            train.add_argument("--steps-per-epoch", type=int, default=self._defaults.get("steps_per_epoch", None), required="steps_per_epoch" not in self._defaults, help="The number of steps per epoch to use for training.")
        if self._use_optional_training:
            train.add_argument("--train", action="store_true", help="Train the model.")
        if self._use_multiprocessing:
            train.add_argument("--workers", type=int, default=self._defaults.get("workers", 0), help="The number of multi-processing workers to use for data generation.")

        #Validation Settings
        val = self.validation_argument_parser
        val.add_argument("--val-batch-size", type=int, default=self._defaults.get("val_batch_size", None), help="The validation batch size to use.")
        if self._use_steps:
            val.add_argument("--val-steps-per-epoch", type=int, default=self._defaults.get("val_steps_per_epoch", None), required="val_steps_per_epoch" not in self._defaults, help="The number of steps per epoch to use for validation.")

    def _init(self):
        config = self.context.config
        if config.epochs == float("inf") or config.epochs is None:
            config.epochs = 2**32 - 1
        if config.val_batch_size is None:
            config.val_batch_size = config.batch_size
        if config.steps_per_epoch is not None and config.val_steps_per_epoch is None:
            config.val_steps_per_epoch = config.config.steps_per_epoch
        if self.context.is_using(dls.module.Wandb):
            wandb = self.context.get(dls.module.Wandb)
            if config.initial_epoch is None:
                config.initial_epoch = wandb.run.step
            wandb.exclude_config_keys([
                "epochs",
                "initial_epoch"
            ])
