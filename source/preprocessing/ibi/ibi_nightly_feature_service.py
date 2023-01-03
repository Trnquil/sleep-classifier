from source.data_services.data_service import DataService
from source.analysis.setup.feature_type import FeatureType
from source.preprocessing.ibi.ibi_feature_service import IbiFeatureService
from source.data_services.dataset import DataSet
from source.data_services.data_loader import DataLoader

import numpy as np
import pandas as pd

class IbiNightlyFeatureService(object):
    
    @staticmethod
    def build_feature_dict_from_cropped(subject_id, session_id, sleep_wake, dataset):
        
        ibi_feature_cropped = DataService.load_feature_raw(subject_id, session_id, FeatureType.cropped_ibi, sleep_wake, dataset)
        ibi_feature_cropped = ibi_feature_cropped*1000
        
        features_dict = IbiFeatureService.get_features(ibi_feature_cropped[:, 1])
        merged_dict = features_dict
                
        
        return merged_dict
    
    
    @staticmethod
    def build_feature_dict_from_epoched(subject_id, session_id, sleep_wake, dataset, cluster_timestamps):
        ibi_feature = DataLoader.load_epoched(subject_id, session_id, FeatureType.epoched_ibi, sleep_wake, dataset)
        ibi_feature = pd.merge(cluster_timestamps, ibi_feature, how="inner", on=["epoch_timestamp"])
        ibi_feature = ibi_feature.drop(columns=['epoch_timestamp'])
        ibi_feature_avg = pd.DataFrame(np.mean(ibi_feature, axis=0))
        ibi_feature_dict = ibi_feature_avg.to_dict()[0]
        return ibi_feature_dict
    
    @staticmethod
    def build_feature_dict_from_epoched_ppg(subject_id, session_id, sleep_wake, dataset, cluster_timestamps):
        ibi_feature = DataLoader.load_epoched(subject_id, session_id, FeatureType.epoched_ibi_from_ppg, sleep_wake, dataset)
        ibi_feature = pd.merge(cluster_timestamps, ibi_feature, how="inner", on=["epoch_timestamp"])
        ibi_feature = ibi_feature.drop(columns=['epoch_timestamp'])
        ibi_feature_avg = pd.DataFrame(np.mean(ibi_feature, axis=0))
        ibi_feature_dict = ibi_feature_avg.to_dict()[0]
        ibi_feature_dict = {'ppg_' + str(key): val for key, val in ibi_feature_dict.items()} 
        return ibi_feature_dict
    
        