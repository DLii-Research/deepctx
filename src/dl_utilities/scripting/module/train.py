from ..context import Context, ContextModule
from ... import scripting as dls

class Train(ContextModule):

    NAME = "Training Hyperparameters"

    def __init__(self, context: Context):
        super().__init__(context)
        self._argument_parser = self.context.argument_parser.add_argument_group(
            title=Train.NAME,
            description="Configuration for the training module.")

    @property
    def argument_parser(self) -> dls.ArgumentParser:
        """
        Get the argument parser for this module.
        """
        return self._argument_parser

    @property
    def initial_epoch(self) -> int:
        if self.context.is_using(dls.module.Wandb):
            wandb = self.context.get(dls.module.Wandb)
            if wandb.run.resumed:
                return wandb.run.step
        return self.context.config.initial_epoch

    def _define_arguments(self):
        """
        Descriptions pulled from W&B documentation:
        https://docs.wandb.ai/ref/python/init
        """
        group = self.argument_parser
        group.add_argument("--epochs", type=str, default=1, help="The number of epochs to train for.")
        group.add_argument("--initial-epoch", type=str, default=0, help="The initial training epoch to start at.")
        group.add_argument("--batch-size", type=str, default=32, help="The training batch size to use.")

    def _init(self):
        if self.context.is_using(dls.module.Wandb):
            wandb = self.context.get(dls.module.Wandb)
            wandb.exclude_config_keys([
                "epochs",
                "initial_epoch"
            ])
