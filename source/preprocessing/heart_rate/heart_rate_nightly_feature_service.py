from source.data_services.data_service import DataService
from source.analysis.setup.feature_type import FeatureType


import numpy as np
import pandas as pd

class HeartRateNightlyFeatureService(object):
    
    @staticmethod
    def build_feature_dict(subject_id, session_id):
        
        heart_rate_feature_raw = DataService.load_feature_raw(subject_id, session_id, FeatureType.cropped_heart_rate)
        
        features_dict = HeartRateNightlyFeatureService.build_var_features(heart_rate_feature_raw[:,1])
        merged_dict = features_dict
                
        
        return merged_dict
    
    
    @staticmethod
    def build_var_features(heart_rate_feature):
        
        hr_std = np.std(heart_rate_feature)
        hr_mean = np.mean(heart_rate_feature)
        
        features_dictionary = {'hr_std': [hr_std], 'hr_mean': [hr_mean]}
        
        return features_dictionary
        