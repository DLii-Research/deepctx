import unittest

# Load Tensorflow mocking
from ....mock import mock_tensorflow as mock_tf

from deepctx.integration import tensorflow as tfi

class TestCpuList(unittest.TestCase):
    def test_cpu_list(self):
        tfi.devices.cpu_list()
        assert mock_tf.config.list_physical_devices.called_with("CPU")

class TestGpuList(unittest.TestCase):
    def test_gpu_list(self):
        tfi.devices.cpu_list()
        assert mock_tf.config.list_physical_devices.called_with("GPU")

# class TestBestGpus(unittest.TestCase):
#     def setUp(self):
#         mock_tf.config.mocking.set_physical_devices([
#             mock_tf.config.PhysicalDevice("GPU", 0),
#             mock_tf.config.PhysicalDevice("GPU", 1),
#         ])

#     def tearDown(self):
#         pass

#     def test_best_gpu(self):
#         tfi.devices.cpu_list()
#         assert mock_tf.config.list_physical_devices.called_with("CPU")
