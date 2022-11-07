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
from hrvanalysis import *

class IbiFeatureService(object):
    # This controlls what ratio of ibi data there must be inside of a 10-minute window to be accepted 
    DataRatio = 0.95

    @staticmethod
    @dispatch(str, str, object)
    def build_hr_features(subject_id, session_id, valid_epochs):
        ibi_collection = DataLoader.load_cropped(subject_id, session_id, FeatureType.cropped_ibi)
        return IbiFeatureService.build_from_collection(ibi_collection, valid_epochs)
    
    @staticmethod
    @dispatch(str, object)
    def build_hr_features(subject_id, valid_epochs):
        ibi_feature = DataService.load_feature_raw(subject_id, FeatureType.cropped_ibi)
        ibi_collection = Collection(subject_id=subject_id, data=ibi_feature)
        return IbiFeatureService.build_from_collection(ibi_collection, valid_epochs)


    @staticmethod
    def build_from_collection(ibi_collection, valid_epochs):
        ibi_features = []

        i = 0
        for epoch in valid_epochs:
            indices_in_range = FeatureService.get_window(ibi_collection.timestamps, epoch)
            
            ibi_values_in_range = ibi_collection.values[indices_in_range].squeeze()
            ibi_timestamps_in_range = ibi_collection.timestamps[indices_in_range].squeeze()
            
            ibi_values_delta = np.sum(ibi_values_in_range)
            ibi_timestamps_delta = ibi_timestamps_in_range[-1] - ibi_timestamps_in_range[0]
            
            x = ibi_values_delta/ibi_timestamps_delta 
            if(ibi_values_delta/ibi_timestamps_delta < IbiFeatureService.DataRatio):
                continue
            
            # We need the IBI values in milliseconds, not seconds
            ibi_values = ibi_values_in_range*1000
            feature_dict = IbiFeatureService.get_features(ibi_values)
                
            epoch_timestamp_dict = {'epoch_timestamp': epoch.timestamp}
            feature_dict = epoch_timestamp_dict | feature_dict
                        
            if(i == 0):
                ibi_features = np.full((len(valid_epochs),len(list(feature_dict.keys()))),np.nan)
                
            ibi_features[i,:] = np.array(list(feature_dict.items()))[:,1]
                
            i += 1
            
        ibi_features = ibi_features[~np.isnan(ibi_features).any(axis=1), :]
        ibi_dataframe = pd.DataFrame(ibi_features, columns=np.array(list(feature_dict.items()))[:,0])
        return ibi_dataframe

    @staticmethod
    def get_features(ibi_values):
        
        feature_dict = get_frequency_domain_features(ibi_values) | get_time_domain_features(ibi_values)
        return feature_dict

