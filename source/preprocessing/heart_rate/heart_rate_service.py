import sys
# caution: path[0] is reserved for script path (or '' in REPL)
# We are inserting the sleepclassifier as a module 
sys.path.insert(1, '../../..')


import numpy as np
import pandas as pd
import os

from source import utils
from source.constants import Constants
from source.preprocessing.collection import Collection
from source.preprocessing.path_service import PathService
from source.analysis.setup.feature_type import FeatureType


class HeartRateService(object):

    @staticmethod
    def load_raw(subject_id):
        raw_hr_path = PathService.get_raw_file_path(subject_id, FeatureType.raw_hr)
        
        #delimiter doesn't do anything but I don't want to break the rest of the program
        heart_rate_array = HeartRateService.load(raw_hr_path, ",")
        heart_rate_array = utils.remove_repeats(heart_rate_array)
        return Collection(subject_id=subject_id, data=heart_rate_array)
    
    #delimiter doesn't do anything but I don't want to break the rest of the program
    @staticmethod
    def load(hr_file, delimiter=" "):
        heart_rate_array = pd.read_csv(str(hr_file), delimiter=delimiter)
        
        #unix start time of the data
        start_time = heart_rate_array.columns.values.tolist()
        start_time = float(start_time[0])
        
        #Let us center the time so that time values aren't too big
        start_time = start_time - Constants.TIME_CENTER
        
        heart_rate_array = heart_rate_array.values.flatten()
        
        #data value frequency in Hertz
        data_frequency = heart_rate_array[0]

        heart_rate_array = heart_rate_array[1:]
        
        timestamps = [(start_time + i*(1/data_frequency)) for i in range(len(heart_rate_array))]
        heart_rate_array = np.stack([timestamps, heart_rate_array], axis=1)
        
        return heart_rate_array

    @staticmethod
    def crop(heart_rate_collection, interval):
        subject_id = heart_rate_collection.subject_id
        timestamps = heart_rate_collection.timestamps
        valid_indices = ((timestamps >= interval.start_time)
                         & (timestamps < interval.end_time)).nonzero()[0]

        cropped_data = heart_rate_collection.data[valid_indices, :]
        return Collection(subject_id=subject_id, data=cropped_data)
