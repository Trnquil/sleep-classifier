import sys
sys.path.insert(1, '../..')

from source.constants import Constants
from source.preprocessing.ibi.ibi_feature_service import IbiFeatureService
from source.preprocessing.activity_count.activity_count_feature_service import ActivityCountFeatureService
from source.preprocessing.raw_data_processor import RawDataProcessor
from source.preprocessing.time.time_based_feature_service import TimeBasedFeatureService
from source.analysis.setup.feature_type import FeatureType
from source.data_services.data_writer import DataWriter
from source.preprocessing.built_service import BuiltService
from source.preprocessing.path_service import PathService

import numpy as np
import pandas as pd



class FeatureBuilder(object):

    @staticmethod
    def build(subject_id):
        
        valid_epochs = RawDataProcessor.get_valid_epochs(subject_id)
        
        count_feature_subject = ActivityCountFeatureService.build_count_feature(subject_id, valid_epochs)
        count_std = np.std(count_feature_subject.iloc[:,1:])
        
        ibi_features_subject = IbiFeatureService.build_hr_features(subject_id, valid_epochs)
        ibi_mean = np.mean(ibi_features_subject.iloc[:,1:], axis=0)
        ibi_std = np.std(ibi_features_subject.iloc[:,1:], axis=0)
        
        # If there is nothing inside ibi_features_subject, we return
        if not np.any(ibi_features_subject):
            return
        
        # Merging all features together
        features_df = pd.merge(count_feature_subject, ibi_features_subject, how="inner", on=["epoch_timestamp"])
        
        sleepsessions = BuiltService.get_built_sleepsession_ids(subject_id, Constants.CROPPED_FILE_PATH)
        for session_id in sleepsessions:
    
            if Constants.VERBOSE:
                print("Building epoched features " + str(subject_id) + "-" + str(session_id) + "...")
                
            valid_epochs = RawDataProcessor.get_valid_epochs(subject_id, session_id)
            
            count_feature = ActivityCountFeatureService.build_count_feature(subject_id, session_id, valid_epochs)
            
            # Normalizing count feature, the first row is a timestamp
            count_feature.iloc[:,1:] = count_feature.iloc[:,1:]/count_std
            
            # Normalizing ibi features, the first row is a timestamp
            ibi_features = IbiFeatureService.build_hr_features(subject_id, session_id, valid_epochs)
            ibi_features.iloc[:,1:] = (ibi_features.iloc[:,1:] - ibi_mean)/ibi_std      
            
            # If there is nothing inside ibi_features, we continue with the next loop
            if not np.any(ibi_features):
                continue
            
            # merging all features together
            features_df = pd.merge(count_feature, ibi_features, how="inner", on=["epoch_timestamp"]).fillna(0)
    
            
            # Writing features to disk
            if(np.any(features_df)):
                # Create needed folders if they don't already exist
                PathService.create_epoched_file_path(subject_id, session_id)
                DataWriter.write_epoched(subject_id, session_id, features_df, FeatureType.epoched)

        
