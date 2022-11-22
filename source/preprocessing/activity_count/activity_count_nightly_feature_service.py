import sys
sys.path.insert(1, '../..')

from source.data_services.data_loader import DataLoader
from source.analysis.setup.feature_type import FeatureType
from source.data_services.dataset import DataSet


import numpy as np
import pandas as pd

class ActivityCountNightlyFeatureService(object):
    
    @staticmethod
    def build_feature_dict(subject_id, session_id, common_timestamps):
        
        count_feature = DataLoader.load_epoched(subject_id, session_id, FeatureType.epoched_count, DataSet.usi)
        count_feature = pd.merge(common_timestamps, count_feature, how="inner", on=["epoch_timestamp"])
        count_feature = count_feature.to_numpy()
        
        features_dict = ActivityCountNightlyFeatureService.build_count_features(count_feature[:,1])
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