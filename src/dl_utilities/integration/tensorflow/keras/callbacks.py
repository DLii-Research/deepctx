import tensorflow as tf
import time

from ... import slurm

class TimeLimitCallback(tf.keras.callbacks.Callback):
    """
    Callback for terminating training before the SLURM job times out.
    """
    def __init__(self, time_limit: float):
        """
        The timestamp in seconds when the job should be terminated.
        """
        super().__init__()
        assert time_limit > time.time()
        self.is_first_epoch = True
        self.prev_epoch_end_time = 0.0
        self.longest_epoch_duration = 0.0
        self.time_limit = time_limit

    def on_train_begin(self, logs=None):
        self.is_first_epoch = True
        self.prev_epoch_and_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        epoch_duration = time.time() - self.prev_epoch_end_time
        # Only keep track of longest epoch duration after the first epoch.
        # This is because the first epoch often takes considerably longer
        # and would not be a good estimate for the remaining epochs.
        longest_epoch_duration = max(epoch_duration, self.longest_epoch_duration)
        if not self.is_first_epoch:
            self.longest_epoch_duration = longest_epoch_duration
            self.is_first_epoch = False
        if time.time() + longest_epoch_duration > self.time_limit:
            self.model.stop_training = True
        self.prev_epoch_end_time = time.time()


class SlurmTimeLimitCallback(tf.keras.callbacks.Callback):
    """
    A callback for terminating training before the SLURM job times out.
    """
    def __init__(self, early_timeout_seconds: float = 120.0):
        if not slurm.is_job():
            time_limit = float("inf")
            print("Not running on SLURM. Time limit is set to infinity.")
        else:
            time_limit = slurm.job_info()["EndTime"].timestamp() - early_timeout_seconds
        super().__init__(time_limit)
