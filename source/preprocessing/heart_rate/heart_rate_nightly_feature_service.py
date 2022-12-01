import sys
sys.path.insert(1, '../../..')

from source.data_services.data_service import DataService
from source.analysis.setup.feature_type import FeatureType
from source.data_services.dataset import DataSet
from source.data_services.data_loader import DataLoader


import numpy as np
import pandas as pd

class HeartRateNightlyFeatureService(object):
    
    @staticmethod
    def build_feature_dict_from_cropped_normalized(subject_id, session_id):
        
        heart_rate_feature_raw = DataService.load_feature_raw(subject_id, session_id, FeatureType.normalized_hr, DataSet.usi)
        
        features_dict = HeartRateNightlyFeatureService.build_var_features(heart_rate_feature_raw[:,1])
        merged_dict = features_dict
        
        merged_dict = {'normalized_' + str(key): val for key, val in merged_dict.items()}          
        
        return merged_dict
    
    
    @staticmethod
    def build_var_features(heart_rate_feature):
        
        hr_std = np.std(heart_rate_feature)
        hr_mean = np.mean(heart_rate_feature)
        
        features_dictionary = {'hr_std': [hr_std], 'hr_mean': [hr_mean]}
        
        return features_dictionary
    
    @staticmethod
    def build_feature_dict_from_epoched(subject_id, session_id, cluster_timestamps):
        
        heart_rate_feature = DataLoader.load_epoched(subject_id, session_id, FeatureType.epoched_hr ,DataSet.usi)
        heart_rate_feature = pd.merge(cluster_timestamps, heart_rate_feature, how="inner", on=["epoch_timestamp"])
        heart_rate_feature = heart_rate_feature.drop(columns=['epoch_timestamp'])
        heart_rate_feature_avg = pd.DataFrame(np.mean(heart_rate_feature, axis=0))
        heart_rate_feature_avg.to_dict()[0]
        return heart_rate_feature_avg.to_dict()[0]
