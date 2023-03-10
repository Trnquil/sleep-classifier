import numpy as np

from source.preprocessing.interval import Interval


class Collection(object):
    
    def __init__(self, subject_id, data, data_frequency):
        self.subject_id = subject_id
        self.data = data
        self.timestamps = data[:, 0]
        self.values = data[:, 1:]
        self.data_frequency = data_frequency

    def get_interval(self):
        return Interval(start_time=np.amin(self.data[:, 0]),
                        end_time=np.amax(self.data[:, 0]))
