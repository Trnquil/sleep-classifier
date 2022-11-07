from source.data_services.data_service import DataService
from source.analysis.setup.feature_type import FeatureType
from source.preprocessing.ibi_feature_service import IbiFeatureService

import numpy as np
import pandas as pd

class HeartRateNightlyFeatureService(object):
    
    @staticmethod
    def build_feature_dict(subject_id, session_id):
        
        ibi_feature_cropped = DataService.load_feature_raw(subject_id, session_id, FeatureType.cropped_ibi)
        ibi_feature_cropped = ibi_feature_cropped*1000
        
        features_dict = IbiFeatureService.get_features(ibi_feature_cropped[:, 1])
        merged_dict = features_dict
                
        
        return merged_dict
    
        