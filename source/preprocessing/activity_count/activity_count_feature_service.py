import numpy as np
import pandas as pd
import os

from source import utils
from source.constants import Constants
from source.preprocessing.activity_count.activity_count_service import ActivityCountService
from source.preprocessing.epoch import Epoch
from source.preprocessing.path_service import PathService
from source.analysis.setup.feature_type import FeatureType
from source.data_service import DataService
from source.preprocessing.feature_service import FeatureService


class ActivityCountFeatureService(object):
    WINDOW_SIZE = 10 * 30 - 15

    @staticmethod
    def get_window(timestamps, epoch):
        start_time = epoch.timestamp - ActivityCountFeatureService.WINDOW_SIZE
        end_time = epoch.timestamp + Epoch.DURATION + ActivityCountFeatureService.WINDOW_SIZE
        timestamps_ravel = timestamps.ravel()
        indices_in_range = np.unravel_index(np.where((timestamps_ravel > start_time) & (timestamps_ravel < end_time)),
                                            timestamps.shape)
        return indices_in_range[0][0]

    @staticmethod
    def build(subject_id, session_id, valid_epochs):
        activity_count_collection = DataService.load_cropped(subject_id, session_id, FeatureType.cropped_count)
        return ActivityCountFeatureService.build_from_collection(activity_count_collection, valid_epochs)

    @staticmethod
    def build_from_collection(activity_count_collection, valid_epochs):
        count_features = []

        activity_count_collection = FeatureService.interpolate(activity_count_collection)
        
        interpolated_timestamps = activity_count_collection.timestamps
        interpolated_counts = activity_count_collection.values

        for epoch in valid_epochs:
            indices_in_range = ActivityCountFeatureService.get_window(interpolated_timestamps, epoch)
            activity_counts_in_range = interpolated_counts[indices_in_range]

            feature = ActivityCountFeatureService.get_feature(activity_counts_in_range)
            count_features.append(feature)

        return np.array(count_features)

    @staticmethod
    def get_feature(count_values):
        convolution = utils.smooth_gauss(count_values.flatten(), np.shape(count_values.flatten())[0])
        return np.array([convolution])
