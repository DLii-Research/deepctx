import tensorflow as tf
from typing import Optional
from unittest import mock

# Mock the module
import sys
sys.modules["tensorflow"] = sys.modules[__name__]

# config module ------------------------------------------------------------------------------------

distribute = tf.distribute
keras = tf.keras

class config:
    class PhysicalDevice:
        def __init__(self, type: str, index: int):
            self.type = type
            self.index = index

    class experimental:
        set_memory_growth = mock.Mock()

    class mocking:
        _physical_devices: list["config.PhysicalDevice"] = []
        _visible_devices: list["config.PhysicalDevice"] = []

        @staticmethod
        def set_physical_devices(devices: list["config.PhysicalDevice"]):
            config.mocking._physical_devices.clear()
            config.mocking._physical_devices.extend(devices)

        @staticmethod
        def set_visible_devices(devices: list["config.PhysicalDevice"]):
            config.mocking._visible_devices.clear()
            config.mocking._visible_devices.extend(devices)

    list_physical_devices = mock.Mock(return_value=mocking._physical_devices)
    get_visible_devices = mock.Mock(return_value=mocking._visible_devices)
    set_visible_devices = mock.Mock()
