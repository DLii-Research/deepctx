from .. import ArgumentParser, wandb
from ..context import Context, ContextModule

class Train(ContextModule):

    NAME = "Training Hyperparameters"

    def __init__(self, context: Context):
        super().__init__(context)
        self._argument_parser = self.context.argument_parser.add_argument_group(
            title=Train.NAME,
            description="Configuration for the training module.")

    @property
    def argument_parser(self) -> ArgumentParser:
        """
        Get the argument parser for this module.
        """
        return self._argument_parser

    def _define_arguments(self):
        """
        Descriptions pulled from W&B documentation:
        https://docs.wandb.ai/ref/python/init
        """
        group = self.argument_parser
        group.add_argument("--epochs", type=str, required=False, default=1, help="The number of epochs to train for.")
        group.add_argument("--batch-size", type=str, required=False, default=32, help="The training batch size to use.")

    def _init(self):
        if self.context.is_using(wandb):
            self.context.get(wandb).exclude_config_keys(["epochs"])
