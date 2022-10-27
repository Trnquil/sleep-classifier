import numpy as np
import pandas as pd
import os

from source import utils
from source.constants import Constants
from source.preprocessing.path_service import PathService
from source.data_services.data_service import DataService
from source.analysis.setup.feature_type import FeatureType
from source.preprocessing.feature_service import FeatureService
from source.data_services.data_loader import DataLoader
from source.preprocessing.collection import Collection

from multipledispatch import dispatch

class HeartRateFeatureService(object):

    @staticmethod
    @dispatch(str, str, object)
    def build(subject_id, session_id, valid_epochs):
        heart_rate_collection = DataLoader.load_cropped(subject_id, session_id, FeatureType.cropped_heart_rate)
        return HeartRateFeatureService.build_from_collection(heart_rate_collection, valid_epochs)
    
    @staticmethod
    @dispatch(str, object)
    def build(subject_id, valid_epochs):
        heart_rate_feature = DataService.load_feature_raw(subject_id, FeatureType.cropped_heart_rate)
        heart_rate_collection = Collection(subject_id=subject_id, data=heart_rate_feature)
        return HeartRateFeatureService.build_from_collection(heart_rate_collection, valid_epochs)


    @staticmethod
    def build_from_collection(heart_rate_collection, valid_epochs):
        heart_rate_features = []

        heart_rate_collection = FeatureService.interpolate(heart_rate_collection)
        heart_rate_collection = FeatureService.convolve(heart_rate_collection)
        
        interpolated_timestamps = heart_rate_collection.timestamps
        interpolated_hr = heart_rate_collection.values

        for epoch in valid_epochs:
            indices_in_range = FeatureService.get_window(interpolated_timestamps, epoch)
            heart_rate_values_in_range = interpolated_hr[indices_in_range]

            feature = HeartRateFeatureService.get_feature(heart_rate_values_in_range)

            heart_rate_features.append(feature)

        return np.array(heart_rate_features)

    @staticmethod
    def get_feature(heart_rate_values):
        return [np.std(heart_rate_values)]

