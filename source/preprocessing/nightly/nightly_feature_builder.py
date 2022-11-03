import sys
sys.path.insert(1, '../..')

from source.constants import Constants
from source.preprocessing.path_service import PathService
from source.preprocessing.clustering.clustering_nightly_feature_service import ClusteringNightlyFeatureService
from source.preprocessing.heart_rate.heart_rate_nightly_feature_service import HeartRateNightlyFeatureService
from source.data_services.data_service import DataService
from source.analysis.setup.feature_type import FeatureType
from source.analysis.setup.subject_builder import SubjectBuilder
from source.data_services.data_service import DataService
from source.preprocessing.built_service import BuiltService
from source.data_services.data_writer import DataWriter
from source.constants import Constants

import pandas as pd
import numpy as np




class NightlyFeatureBuilder(object):

    @staticmethod
    def build():

        if Constants.VERBOSE:
            print("Building nightly features...")
            
        i = 0
        for subject_id in BuiltService.get_built_subject_ids(Constants.CROPPED_FILE_PATH):
            for session_id in BuiltService.get_built_sleepsession_ids(subject_id, Constants.CROPPED_FILE_PATH):
                feature_dict = NightlyFeatureBuilder.build_feature_dict(subject_id, session_id)
                
                feature_row = pd.DataFrame(feature_dict)
                
                if i == 0:
                    nightly_dataframe = feature_row
                else:    
                    nightly_dataframe = pd.concat([nightly_dataframe, feature_row], axis=0)
                
                i += 1

        
        DataWriter.write_nightly(nightly_dataframe)
        
    @staticmethod
    def build_feature_dict(subject_id, session_id):
        
        subject_session_dict = {'subject_id': subject_id, 'session_id': session_id}
        clustering_features_dict = ClusteringNightlyFeatureService.build_feature_dict(subject_id, session_id)
        
        sleepquality_avg = np.mean(DataService.load_feature_raw(subject_id, FeatureType.sleep_quality))
        sleepquality = DataService.load_feature_raw(subject_id, session_id, FeatureType.sleep_quality)
        sleepquality = 0 if sleepquality < sleepquality_avg else 1
        sleepquality_dict = {'sleep_quality': sleepquality}
        
        heart_rate_features_dict = HeartRateNightlyFeatureService.build_feature_dict(subject_id, session_id)
        
        merged_dict = subject_session_dict | clustering_features_dict | heart_rate_features_dict | sleepquality_dict
        
        return merged_dict

        
