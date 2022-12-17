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
from source.data_services.dataset import DataSet
from source.preprocessing.usi_raw_data_processor import UsiRawDataProcessor


from multipledispatch import dispatch
import torch

class HeartRateFeatureService(object):
    
    @staticmethod
    @dispatch(str, str, object)
    def build(subject_id, session_id, dataset):
        heart_rate_collection = DataLoader.load_cropped(subject_id, session_id, FeatureType.cropped_hr, dataset)
        valid_epochs = UsiRawDataProcessor.get_valid_epochs([heart_rate_collection])
        return HeartRateFeatureService.build_from_collection(heart_rate_collection, valid_epochs)
    
    @staticmethod
    @dispatch(str, object)
    def build(subject_id, dataset):
        heart_rate_feature = DataService.load_feature_raw(subject_id, FeatureType.cropped_hr, dataset)
        heart_rate_collection = Collection(subject_id=subject_id, data=heart_rate_feature, data_frequency=0)
        valid_epochs = UsiRawDataProcessor.get_valid_epochs([heart_rate_collection])
        return HeartRateFeatureService.build_from_collection(heart_rate_collection, valid_epochs)


    @staticmethod
    def build_from_collection(heart_rate_collection, valid_epochs):
        heart_rate_features = np.zeros((0,3))

        #heart_rate_collection = FeatureService.interpolate(heart_rate_collection)
        #heart_rate_collection = FeatureService.convolve(heart_rate_collection)
        
        interpolated_timestamps = heart_rate_collection.timestamps
        interpolated_hr = heart_rate_collection.values

        for epoch in valid_epochs:
            indices_in_range = FeatureService.get_window(interpolated_timestamps, epoch)
            heart_rate_values_in_range = interpolated_hr[indices_in_range]

            hr_mean, hr_std = HeartRateFeatureService.get_feature(heart_rate_values_in_range)

            heart_rate_features = np.concatenate([heart_rate_features, np.array([epoch.timestamp, hr_mean, hr_std]).reshape(1,3)], axis=0)
        
        hr_features_df = pd.DataFrame(heart_rate_features)
        hr_features_df.columns = ["epoch_timestamp", "hr_mean", "hr_std"]
        return hr_features_df
    
    def build_GEMINI_clusters(heart_rate_collection, valid_epochs, model):
        clusters = np.zeros((0,2))
        heart_rate_collection = FeatureService.interpolate(heart_rate_collection)
        
        interpolated_timestamps = heart_rate_collection.timestamps
        interpolated_hr = heart_rate_collection.values

        for epoch in valid_epochs:
            indices_in_range = FeatureService.get_window(interpolated_timestamps, epoch)
            heart_rate_values_in_range = interpolated_hr[indices_in_range]
            
            if(len(heart_rate_values_in_range) >= 500):
                hr_values = np.expand_dims(heart_rate_values_in_range[:500].squeeze(), axis = (0,1))
                cluster_onehot = model(torch.tensor(hr_values).float())
                cluster = cluster_onehot[0].detach().numpy().argmax()         

                clusters = np.concatenate([clusters, np.array([epoch.timestamp, cluster]).reshape(1,2)], axis=0)
        
        clusters_df = pd.DataFrame(clusters)
        clusters_df.columns = ["epoch_timestamp", "cluster"]
        return clusters_df

    @staticmethod
    def get_feature(heart_rate_values):
        return np.mean(heart_rate_values), np.std(heart_rate_values)

