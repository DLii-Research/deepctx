import numpy as np
import random
from ..context import ContextModule
from ... import scripting as dls

class Rng(ContextModule):

    NAME = "Random Number Generator"

    def rng(self) -> np.random.Generator:
        return self._rng

    def _define_arguments(self):
        group = self.context.argument_parser.add_argument_group(
            title=Rng.NAME,
            description="Configuration for the random number generator.")
        group.add_argument("--seed", type=int, default=None)

    def _init(self):
        random.seed(self.context.config.seed)
        np.random.seed(self.context.config.seed)
        if self.context.is_using(dls.module.Tensorflow):
            import tensorflow as tf
            tf.random.set_seed(self.context.config.seed)
        self._rng = np.random.default_rng(self.context.config.seed)
