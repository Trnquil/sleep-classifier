import sys
sys.path.insert(1, '../..')

from source.constants import Constants
from source.preprocessing.activity_count.activity_count_feature_service import ActivityCountFeatureService
from source.preprocessing.heart_rate.heart_rate_feature_service import HeartRateFeatureService
from source.preprocessing.raw_data_processor import RawDataProcessor
from source.preprocessing.time.time_based_feature_service import TimeBasedFeatureService
from source.analysis.setup.feature_type import FeatureType
from source.data_services.data_writer import DataWriter
from source.preprocessing.built_service import BuiltService

import numpy as np



class FeatureBuilder(object):

    @staticmethod
    def build(subject_id):
        
        valid_epochs = RawDataProcessor.get_valid_epochs(subject_id)
        
        count_feature_subject = ActivityCountFeatureService.build(subject_id, valid_epochs)
        count_std = np.std(count_feature_subject)
        
        heart_rate_feature_subject = HeartRateFeatureService.build(subject_id, valid_epochs)
        hr_mean = np.mean(heart_rate_feature_subject, axis=0)
        hr_std = np.std(heart_rate_feature_subject, axis=0)
        
        sleepsessions = BuiltService.get_built_sleepsession_ids(subject_id)
        for session_id in sleepsessions:
            if Constants.VERBOSE:
                print("Getting valid epochs...")
            valid_epochs = RawDataProcessor.get_valid_epochs(subject_id, session_id)
    
            if Constants.VERBOSE:
                print("Building features...")
                
            count_feature = ActivityCountFeatureService.build(subject_id, session_id, valid_epochs)
            
            # Normalizing count feature
            count_feature = count_feature/count_std
            
            # Normalizing heart rate feature
            heart_rate_feature = HeartRateFeatureService.build(subject_id, session_id, valid_epochs)
            heart_rate_feature = (heart_rate_feature - hr_mean)/hr_std
            
            if Constants.INCLUDE_CIRCADIAN:
                circadian_feature = TimeBasedFeatureService.build_circadian_model(subject_id, session_id, valid_epochs)
                TimeBasedFeatureService.write_circadian_model(subject_id, session_id, circadian_feature)
    
            cosine_feature = TimeBasedFeatureService.build_cosine(valid_epochs)
            time_feature = TimeBasedFeatureService.build_time(valid_epochs)
    
    
            # Writing all features to their files
            DataWriter.write_epoched(subject_id, session_id, cosine_feature, FeatureType.epoched_cosine)
            DataWriter.write_epoched(subject_id, session_id, time_feature, FeatureType.epoched_time)
            DataWriter.write_epoched(subject_id, session_id, count_feature, FeatureType.epoched_count)
            DataWriter.write_epoched(subject_id, session_id, heart_rate_feature, FeatureType.epoched_heart_rate)

        
