import argparse
import time
import dl_utilities as dlu
import dl_utilities.scripting as dls

# import tensorflow as tf
from dl_utilities.lazy import tensorflow as tf


class PersistentModel(dls.module.Wandb.PersistentObject[tf.keras.Model]):
    def create(self, context: dls.Context):
        return tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='relu', input_shape=(784,)),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

    def save(self):
        self.instance.save(self.path("model"))

    def load(self):
        return tf.keras.models.load_model(self.path("model"))


def main(context: dls.Context):
    model = PersistentModel().instance


if __name__ == '__main__':
    context = dls.Context(main)
    context.use(dls.module.Tensorflow)
    context.use(dls.module.Wandb)
    context.execute()
