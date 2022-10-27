from source.data_services.data_service import DataService
from source import utils
import numpy as np
from source.preprocessing.collection import Collection
from source.preprocessing.epoch import Epoch

class FeatureService(object):
    WINDOW_SIZE = 10 * 30 - 15
    
    @staticmethod
    def normalize(collection, feature_type):
        subject_id = collection.subject_id
        feature = DataService.load_feature_raw(subject_id, feature_type)[:,1]
        mean = np.mean(feature, axis=0)
        std = np.std(feature, axis=0)
        
        normalized_values = (collection.values - mean)/std
        timestamps = np.expand_dims(collection.timestamps, axis=1)
        normalized_data = np.concatenate((timestamps, normalized_values), axis=1)
        
        return Collection(subject_id= subject_id, data = normalized_data)
    
    @staticmethod 
    def interpolate(collection):
        timestamps = collection.timestamps.flatten()
        values = collection.values.flatten()
        
        interpolated_timestamps = np.arange(np.amin(timestamps),
                                            np.amax(timestamps), 1)
        
        interpolated_values = np.interp(interpolated_timestamps, timestamps, values)

        interpolated_data = np.stack((interpolated_timestamps, interpolated_values), axis=1)
        return Collection(collection.subject_id, interpolated_data)
    
    @staticmethod
    def convolve(collection):
        convolved_values = utils.convolve_with_dog(collection.values.flatten(), FeatureService.WINDOW_SIZE)

        convolved_data = np.stack((collection.timestamps, convolved_values), axis=1)
        return Collection(collection.subject_id, convolved_data)
    
    @staticmethod
    def get_window(timestamps, epoch):
        start_time = epoch.timestamp - FeatureService.WINDOW_SIZE
        end_time = epoch.timestamp + Epoch.DURATION + FeatureService.WINDOW_SIZE
        timestamps_ravel = timestamps.ravel()
        indices_in_range = np.unravel_index(np.where((timestamps_ravel > start_time) & (timestamps_ravel < end_time)),
                                            timestamps.shape)
        return indices_in_range[0][0]