from source.data_service import DataService
from source import utils
import numpy as np

class FeatureService(object):
    
    @staticmethod
    def normalize(collection, feature_type):
        subject_id = collection.subject_id
        feature = DataService.load_feature_raw(subject_id, feature_type)
        mean = np.mean(feature)
        std = np.std(feature)
        
        collection.values = (collection.values - mean)/std
        
        return collection
    
    @staticmethod 
    def interpolate_and_convolve(collection, window_size):
        timestamps = collection.timestamps.flatten()
        values = collection.values.flatten()
        
        interpolated_timestamps = np.arange(np.amin(timestamps),
                                            np.amax(timestamps), 1)
        
        interpolated_values = np.interp(interpolated_timestamps, timestamps, values)

        interpolated_values = utils.convolve_with_dog(interpolated_values, window_size)
        
        return interpolated_timestamps, interpolated_values