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
from source.preprocessing.heart_rate.heart_rate_feature_service import HeartRateFeatureService
from source.data_services.data_service import DataService
from source.preprocessing.collection import Collection
from source.data_services.dataset import DataSet

import numpy as np
import pandas as pd
from multipledispatch import dispatch



class EpochedFeatureBuilder(object):

    @staticmethod
    def build(subject_id):
        valid_epochs = EpochedFeatureBuilder.get_valid_epochs(subject_id)
        
        count_feature_subject = ActivityCountFeatureService.build_count_feature(subject_id, valid_epochs)
        count_std = np.std(count_feature_subject.iloc[:,1:])
        
        ibi_features_subject = IbiFeatureService.build_hr_features(subject_id, valid_epochs)
        ibi_mean = np.mean(ibi_features_subject.iloc[:,1:], axis=0)
        ibi_std = np.std(ibi_features_subject.iloc[:,1:], axis=0)
        
        hr_features_subject = HeartRateFeatureService.build(subject_id, valid_epochs)
        hr_mean = np.mean(hr_features_subject.iloc[:,1:], axis=0)
        hr_std = np.std(hr_features_subject.iloc[:,1:], axis=0)
        
        # If there is nothing inside ibi_features_subject, we return
        if not np.any(ibi_features_subject):
            return
        
        
        sleepsessions = BuiltService.get_built_sleepsession_ids(subject_id, FeatureType.cropped, DataSet.usi)
        for session_id in sleepsessions:
    
            if Constants.VERBOSE:
                print("Building epoched features " + str(subject_id) + "-" + str(session_id) + "...")
                
            valid_epochs = EpochedFeatureBuilder.get_valid_epochs(subject_id, session_id)
            
            count_feature = ActivityCountFeatureService.build_count_feature(subject_id, session_id, valid_epochs)
            
            # Normalizing count feature, the first row is a timestamp
            count_feature.iloc[:,1:] = count_feature.iloc[:,1:]/count_std
            
            # Normalizing ibi features, the first row is a timestamp
            ibi_features = IbiFeatureService.build_hr_features(subject_id, session_id, valid_epochs)
            ibi_features.iloc[:,1:] = (ibi_features.iloc[:,1:] - ibi_mean)/ibi_std      
            
            # If there is nothing inside ibi_features, we continue with the next loop
            if not np.any(ibi_features):
                continue
            
            hr_feature = HeartRateFeatureService.build(subject_id, session_id, valid_epochs)
            hr_feature.iloc[:,1:] = (hr_feature.iloc[:,1:] - hr_mean)/hr_std     
            
            
            # merging all features together
            features_df = pd.merge(count_feature, hr_feature, how="inner", on=["epoch_timestamp"]).fillna(0)
    
            
            # Writing features to disk
            if(np.any(features_df)):
                # Create needed folders if they don't already exist
                PathService.create_epoched_folder_path(subject_id, session_id, DataSet.usi)
                DataWriter.write_epoched(features_df, subject_id, session_id, FeatureType.epoched, DataSet.usi)
                
    @staticmethod
    @dispatch(str)         
    def get_valid_epochs(subject_id):
        motion_feature = DataService.load_feature_raw(subject_id,  FeatureType.cropped_motion, DataSet.usi)
        motion_collection = Collection(subject_id=subject_id, data=motion_feature, data_frequency=0)
        
        ibi_feature = DataService.load_feature_raw(subject_id, FeatureType.cropped_ibi, DataSet.usi)
        ibi_collection = Collection(subject_id=subject_id, data=ibi_feature, data_frequency=0)

        collections = [motion_collection, ibi_collection]
        valid_epochs = RawDataProcessor.get_valid_epochs(collections)
        return valid_epochs
    
    @staticmethod
    @dispatch(str, str)
    def get_valid_epochs(subject_id, session_id):
        motion_feature = DataService.load_feature_raw(subject_id, session_id, FeatureType.cropped_motion, DataSet.usi)
        motion_collection = Collection(subject_id=subject_id, data=motion_feature, data_frequency=0)
        
        ibi_feature = DataService.load_feature_raw(subject_id, session_id, FeatureType.cropped_ibi, DataSet.usi)
        ibi_collection = Collection(subject_id=subject_id, data=ibi_feature, data_frequency=0)

        collections = [motion_collection, ibi_collection]
        valid_epochs = RawDataProcessor.get_valid_epochs(collections)
        return valid_epochs

        
