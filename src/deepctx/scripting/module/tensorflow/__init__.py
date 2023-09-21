import argparse
import os
from ...context import Context, ContextModule
from .... import integration
from .... import scripting as dls
# from ....lazy import tensorflow as tf

class Tensorflow(ContextModule):

    NAME = "Tensorflow"

    def __init__(self, context: Context):
        super().__init__(context)
        self._optimizer_argument_parsers = {}
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    def min_log_level(self, level: str|int) -> "Tensorflow":
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(level)
        return self

    def _define_arguments(self):
        parser = self.context.argument_parser.add_argument_group("Tensorflow Settings")
        parser.add_argument("--num-gpus", type=int, default=0, help="The number of GPUs to use")
        parser.add_argument("--gpu-ids", type=lambda x: list(map(int, x.split(','))), default=None, help="A comma separated list of GPUs to make avaliable for selection.")
        parser.add_argument("--use-static-memory", action="store_true", default=False, help="Use static memory allocation.")

    def _select_gpus(self, config: argparse.Namespace):
        gpu_list = integration.tensorflow.devices.gpu_list()
        if config.gpu_ids is not None:
            gpu_list = [gpu_list[gpu_index] for gpu_index in config.gpu_ids]
            if config.num_gpus == 0:
                config.num_gpus = len(gpu_list)
        if config.num_gpus > 0:
            assert config.num_gpus <= len(gpu_list), f"Cannot use {config.num_gpus} GPUs, only {len(gpu_list)} are available."
            gpu_list = integration.tensorflow.devices.best_gpus(gpu_list, config.num_gpus)
        else:
            gpu_list = []
        print("GPU list being used:", gpu_list)
        integration.tensorflow.devices.use(gpus=gpu_list, use_dynamic_memory=not config.use_static_memory)

    def _init(self):
        config = self.context.config
        self._select_gpus(config)
        if self.context.is_using(dls.module.Wandb):
            wandb = self.context.get(dls.module.Wandb)
            wandb.exclude_config_keys([
                "num-gpus",
                "gpu-ids",
                "use-static-memory"
            ])
