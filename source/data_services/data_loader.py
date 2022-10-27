from source.analysis.setup.feature_type import FeatureType
from source.preprocessing.path_service import PathService
from source.constants import Constants
from source import utils
from source.preprocessing.collection import Collection

import numpy as np
import pandas as pd

class DataLoader(object):
    
    @staticmethod
    def load_raw(subject_id, feature_type):
        raw_path = PathService.get_raw_file_path(subject_id, feature_type)
    
        feature_array = pd.read_csv(str(raw_path), delimiter=",")
        
        #unix start time of the data
        start_time = feature_array.columns.values.tolist()
        start_time = float(start_time[0])
        
        #Let us center the time so that time values aren't too big
        start_time = start_time - Constants.TIME_CENTER
        
        if(feature_array.ndim == 1):
            np.expand_dims(feature_array, axis=1)
        
        feature_array = feature_array.to_numpy()
        
        #data value frequency in Hertz
        data_frequency = feature_array[0][0]

        feature_array = feature_array[1:][:]
        
        # We need to normalize acceleration data
        if(feature_type == FeatureType.raw_acc):
            feature_array = feature_array/64
        
        timestamps = [(start_time + i*(1/data_frequency)) for i in range(len(feature_array))]
        timestamps = np.expand_dims(timestamps, axis=1)
        feature_array = np.concatenate([timestamps, feature_array], axis=1)
        feature_array = utils.remove_repeats(feature_array)
        return Collection(subject_id=subject_id, data=feature_array)
    
    @staticmethod
    def load_cropped(subject_id, session_id, feature_type):
        file_path = PathService.get_cropped_file_path(subject_id, session_id, feature_type)
        values = pd.read_csv(str(file_path), delimiter=" ").values
        return Collection(subject_id=subject_id, data=values)
    
    @staticmethod
    def load_epoched(subject_id, session_id, feature_type):
        feature_path = PathService.get_epoched_file_path(subject_id, session_id, feature_type)
        feature = pd.read_csv(str(feature_path)).values
        return feature
    
    @staticmethod
    def load_nightly():
        nightly_feature_path = PathService.get_nightly_file_path()
        nightly_feature_dataframe = pd.read_csv(str(nightly_feature_path))
        return nightly_feature_dataframe