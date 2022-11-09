import sys
sys.path.insert(1, '../..')

from source.data_services.data_service import DataService
from source.analysis.setup.feature_type import FeatureType


import numpy as np
import pandas as pd

class ActivityCountNightlyFeatureService(object):
    
    @staticmethod
    def build_feature_dict(subject_id, session_id):
        
        count_feature_data = DataService.load_feature_raw(subject_id, session_id, FeatureType.cropped_count)
        
        features_dict = ActivityCountNightlyFeatureService.build_count_features(count_feature_data[:,1])
        merged_dict = features_dict
                
        
        return merged_dict
    
    
    @staticmethod
    def build_count_features(count_feature):
        
        count_feature_len = count_feature.shape[0]
        features_dictionary = {
        'count_p_notzero': [1 - np.sum(count_feature == 0)/count_feature_len],
        'count_mean': [np.mean(count_feature)],
        'count_max': [np.max(count_feature)],
        'count_total': [np.sum(count_feature)],
        'count_longest_movement': [ActivityCountNightlyFeatureService.find_longest_chain(count_feature)]
        }
        
        
        return features_dictionary
        
    @staticmethod
    def find_longest_chain(count_feature):
        count = 0
        max_count = 0
        for i in range(len(count_feature)):
            if count_feature[i] != 0:
                count += 1
            else:
                max_count = count if count > max_count else max_count
                count = 0
        return max_count