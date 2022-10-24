import numpy as np
import pandas as pd
import os

from source import utils
from source.constants import Constants
from source.preprocessing.epoch import Epoch
from source.preprocessing.heart_rate.heart_rate_service import HeartRateService
from source.preprocessing.path_service import PathService
from source.data_service import DataService
from source.analysis.setup.feature_type import FeatureType

class HeartRateFeatureService(object):
    WINDOW_SIZE = 10 * 30 - 15

    @staticmethod
    def build(subject_id, session_id, valid_epochs):
        heart_rate_collection = DataService.load_cropped(subject_id, session_id, FeatureType.cropped_heart_rate)
        return HeartRateFeatureService.build_from_collection(heart_rate_collection, valid_epochs)

    @staticmethod
    def build_from_collection(heart_rate_collection, valid_epochs):
        heart_rate_features = []

        interpolated_timestamps, interpolated_hr = HeartRateFeatureService.interpolate_and_normalize(
            heart_rate_collection)

        for epoch in valid_epochs:
            indices_in_range = HeartRateFeatureService.get_window(interpolated_timestamps, epoch)
            heart_rate_values_in_range = interpolated_hr[indices_in_range]

            feature = HeartRateFeatureService.get_feature(heart_rate_values_in_range)

            heart_rate_features.append(feature)

        return np.array(heart_rate_features)

    @staticmethod
    def get_window(timestamps, epoch):
        start_time = epoch.timestamp - HeartRateFeatureService.WINDOW_SIZE
        end_time = epoch.timestamp + Epoch.DURATION + HeartRateFeatureService.WINDOW_SIZE
        timestamps_ravel = timestamps.ravel()
        indices_in_range = np.unravel_index(np.where((timestamps_ravel > start_time) & (timestamps_ravel < end_time)),
                                            timestamps.shape)
        return indices_in_range[0][0]

    @staticmethod
    def get_feature(heart_rate_values):
        return [np.std(heart_rate_values)]

    @staticmethod
    def interpolate_and_normalize(heart_rate_collection):
        timestamps = heart_rate_collection.timestamps.flatten()
        heart_rate_values = heart_rate_collection.values.flatten()
        interpolated_timestamps = np.arange(np.amin(timestamps),
                                            np.amax(timestamps), 1)
        interpolated_hr = np.interp(interpolated_timestamps, timestamps, heart_rate_values)

        interpolated_hr = utils.convolve_with_dog(interpolated_hr, HeartRateFeatureService.WINDOW_SIZE)

        scalar = np.percentile(np.abs(interpolated_hr), 90)
        interpolated_hr = interpolated_hr / scalar

        return interpolated_timestamps, interpolated_hr
