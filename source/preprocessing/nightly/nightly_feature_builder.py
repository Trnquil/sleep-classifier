import sys
sys.path.insert(1, '../../..')

from source.constants import Constants
from source.preprocessing.path_service import PathService
from source.preprocessing.clustering.cluster_nightly_feature_service import ClusterNightlyFeatureService
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
from source.data_services.data_frame_loader import DataFrameLoader

import pandas as pd
import numpy as np
from sklearn.utils import resample




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
            regex="ibi_.*|count_.*|hr_.*"
            subject_mean = np.mean(subject_dataframe.filter(regex=regex), axis=0)
            subject_std = np.std(subject_dataframe.filter(regex=regex), axis=0)*2
            
            subject_dataframe_normalized = subject_dataframe.copy()
            subject_dataframe_normalized[subject_dataframe_normalized.filter(regex=regex).columns] = (subject_dataframe_normalized.filter(regex=regex) - subject_mean)/subject_std
            subject_dataframe_normalized = subject_dataframe_normalized.fillna(0)
            
            if subject_index == 0:
                nightly_dataframe = subject_dataframe
                nightly_dataframe_normalized = subject_dataframe_normalized
            else:
                nightly_dataframe = pd.concat([nightly_dataframe, subject_dataframe], axis=0)
                nightly_dataframe_normalized = pd.concat([nightly_dataframe_normalized, subject_dataframe_normalized], axis=0)
            
            subject_index += 1
        
        DataWriter.write_nightly(nightly_dataframe, FeatureType.nightly)
        DataWriter.write_nightly(nightly_dataframe_normalized, FeatureType.normalized_nightly)

        
        
    @staticmethod
    def build_feature_dict(subject_id, session_id):
        try:
            clusters = DataFrameLoader.load_feature_dataframe(subject_id, session_id, [FeatureType.cluster], DataSet.usi)
            cluster_timestamps = clusters['epoch_timestamp']
            
            subject_session_dict = {'subject_id': subject_id, 'session_id': session_id}
            
            ibi_features_dict = IbiNightlyFeatureService.build_feature_dict_from_epoched(subject_id, session_id, cluster_timestamps)
            
            count_features_dict = ActivityCountNightlyFeatureService.build_feature_dict_from_epoched(subject_id, session_id, cluster_timestamps)
            
            hr_features_dict = HeartRateNightlyFeatureService.build_feature_dict_from_epoched(subject_id, session_id, cluster_timestamps)
            
            cluster_features_dict = ClusterNightlyFeatureService.build_feature_dict(subject_id, session_id)
            
            sleepquality_avg = np.mean(DataService.load_feature_raw(subject_id, FeatureType.sleep_quality, DataSet.usi))
            sleepquality = DataService.load_feature_raw(subject_id, session_id, FeatureType.sleep_quality, DataSet.usi)
            sleepquality = 0 if sleepquality < sleepquality_avg else 1
            sleepquality_dict = {'sleep_quality': sleepquality}
            
            merged_dict = subject_session_dict | cluster_features_dict | hr_features_dict | count_features_dict | ibi_features_dict | sleepquality_dict
            
            return merged_dict
        except:
            print("Error: ", sys.exc_info()[0], " while building nightly features for " + str(subject_id), ", session " + str(session_id))
        
    @staticmethod 
    def upsample_minority(nightly_dataframe):
        
        df_1 = nightly_dataframe[nightly_dataframe.sleep_quality==1]
        df_0 = nightly_dataframe[nightly_dataframe.sleep_quality==0]
        
        if df_1.shape[0] > df_0.shape[0]:
            df_majority = df_1
            df_minority = df_0
        else:
            df_majority = df_0
            df_minority = df_1
                                        
        df_minority_upsampled = resample(df_minority, 
        replace=True,     # sample with replacement
        n_samples=df_majority.shape[0],    # to match majority class
        random_state=123)
        
        nightly_dataframe = pd.concat([df_majority, df_minority_upsampled])
        return nightly_dataframe

        
