import numpy as np
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TimeGeneration:
    def __init__(self, local_average, local_stddev, local_min, global_average, global_stddev, global_min):
        self.local_average = local_average
        self.local_stddev = local_stddev
        self.local_min = local_min
        self.global_average = global_average
        self.global_stddev = global_stddev
        self.global_min = global_min

    def get_local(self, size):
        return np.maximum(np.random.normal(self.local_average, self.local_stddev, size), self.local_min)

    def get_global(self, size):
        return np.maximum(np.random.normal(self.global_average, self.global_stddev, size), self.global_min)
