import sys
sys.path.insert(1, '../../..')

from source.constants import Constants
from source.preprocessing.path_service import PathService
from source.preprocessing.clustering.clustering_nightly_feature_service import ClusteringNightlyFeatureService
from source.preprocessing.ibi.ibi_nightly_feature_service import IbiNightlyFeatureService
from source.data_services.data_service import DataService
from source.analysis.setup.feature_type import FeatureType
from source.analysis.setup.subject_builder import SubjectBuilder
from source.data_services.data_service import DataService
from source.preprocessing.built_service import BuiltService
from source.data_services.data_writer import DataWriter
from source.constants import Constants
from source.preprocessing.activity_count.activity_count_nightly_feature_service import ActivityCountNightlyFeatureService
from source.preprocessing.heart_rate.heart_rate_nightly_feature_service import HeartRateNightlyFeatureService
from source.data_services.dataset import DataSet
from source.data_services.data_loader import DataLoader

import pandas as pd
import numpy as np




class NightlyFeatureBuilder(object):

    @staticmethod
    def build():

        if Constants.VERBOSE:
            print("Building nightly features...")
            
        subject_index = 0
        for subject_id in BuiltService.get_built_subject_ids(FeatureType.epoched, DataSet.usi):
            
            session_index = 0
            for session_id in BuiltService.get_built_sleepsession_ids(subject_id, FeatureType.epoched, DataSet.usi):
                feature_dict = NightlyFeatureBuilder.build_feature_dict(subject_id, session_id)
                
                feature_row = pd.DataFrame(feature_dict)
                
                if session_index == 0:
                    subject_dataframe = feature_row
                else:    
                    subject_dataframe = pd.concat([subject_dataframe, feature_row], axis=0)
                    
                session_index += 1
            
            #Normalizing features across the subject and filling 0 for features with std 0
            # regex="ibi_.*|count_.*|hr_.*"
            # subject_mean = np.mean(subject_dataframe.filter(regex=regex), axis=0)
            # subject_std = np.std(subject_dataframe.filter(regex=regex), axis=0)*2
            # subject_dataframe[subject_dataframe.filter(regex=regex).columns] = (subject_dataframe.filter(regex=regex) - subject_mean)/subject_std
            # subject_dataframe = subject_dataframe.fillna(0)
            
            if subject_index == 0:
                nightly_dataframe = subject_dataframe
            else:
                nightly_dataframe = pd.concat([nightly_dataframe, subject_dataframe], axis=0)
            
            subject_index += 1
        
        # FOR TESTING PURPOSES:
        DataWriter.write_nightly(nightly_dataframe[['subject_id', 'session_id', 'c_0', 'c_1', 'c_2', 'c_3', 'c_4', 'c_5', 'hr_mean', 'hr_std', 'sleep_quality']])

        
        
    @staticmethod
    def build_feature_dict(subject_id, session_id):
        
        feature_types = [FeatureType.epoched_cluster, FeatureType.epoched_hr]
        common_timestamps = DataLoader.load_common_epoched_timestamps(subject_id, session_id, feature_types, DataSet.usi)
        subject_session_dict = {'subject_id': subject_id, 'session_id': session_id}
        clustering_features_dict = ClusteringNightlyFeatureService.build_feature_dict(subject_id, session_id, common_timestamps)
        
        sleepquality_avg = np.mean(DataService.load_feature_raw(subject_id, FeatureType.sleep_quality, DataSet.usi))
        sleepquality = DataService.load_feature_raw(subject_id, session_id, FeatureType.sleep_quality, DataSet.usi)
        sleepquality = 0 if sleepquality < sleepquality_avg else 1
        sleepquality_dict = {'sleep_quality': sleepquality}
        
        # ibi_features_dict = IbiNightlyFeatureService.build_feature_dict(subject_id, session_id)
        # ibi_features_dict = {'ibi_' + str(key): val for key, val in ibi_features_dict.items()}
        
        count_features_dict = ActivityCountNightlyFeatureService.build_feature_dict(subject_id, session_id, common_timestamps)
        
        hr_features_dict = HeartRateNightlyFeatureService.build_feature_dict_from_epoched(subject_id, session_id, common_timestamps)
        
        merged_dict = subject_session_dict | clustering_features_dict | hr_features_dict | sleepquality_dict
        
        return merged_dict

        
