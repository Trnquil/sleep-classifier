import numpy as np
import pandas as pd
import os

from source import utils
from source.constants import Constants
from source.preprocessing.activity_count.activity_count_service import ActivityCountService
from source.preprocessing.epoch import Epoch
from source.preprocessing.path_service import PathService
from source.analysis.setup.feature_type import FeatureType
from source.data_services.data_service import DataService
from source.preprocessing.feature_service import FeatureService
from source.data_services.data_loader import DataLoader
from source.preprocessing.collection import Collection

from multipledispatch import dispatch


class ActivityCountFeatureService(object):

    @staticmethod
    @dispatch(str, str, object)
    def build_count_feature(subject_id, session_id, valid_epochs):
        activity_count_collection = DataLoader.load_cropped(subject_id, session_id, FeatureType.cropped_count)
        return ActivityCountFeatureService.build_from_collection(activity_count_collection, valid_epochs)
    
    @staticmethod
    @dispatch(str, object)
    def build_count_feature(subject_id, valid_epochs):
        activity_count_feature = DataService.load_feature_raw(subject_id, FeatureType.cropped_count)
        activity_count_collection = Collection(subject_id=subject_id, data=activity_count_feature, data_frequency=0)
        return ActivityCountFeatureService.build_from_collection(activity_count_collection, valid_epochs)

    @staticmethod
    def build_from_collection(activity_count_collection, valid_epochs):
        count_features = []

        activity_count_collection = FeatureService.interpolate(activity_count_collection)
        
        interpolated_timestamps = activity_count_collection.timestamps
        interpolated_counts = activity_count_collection.values

        for epoch in valid_epochs:
            indices_in_range = FeatureService.get_window(interpolated_timestamps, epoch)
            activity_counts_in_range = interpolated_counts[indices_in_range]

            feature = ActivityCountFeatureService.get_feature(activity_counts_in_range)
            count_features.append([epoch.timestamp, feature])

        count_feature_array = np.array(count_features)
        count_feature_df = pd.DataFrame(count_feature_array, columns=["epoch_timestamp","count"])
        return count_feature_df

    @staticmethod
    def get_feature(count_values):
        convolution = utils.smooth_gauss(count_values.flatten(), np.shape(count_values.flatten())[0])
        return convolution
