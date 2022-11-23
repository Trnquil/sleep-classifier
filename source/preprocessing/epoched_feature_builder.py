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
        
        count_feature_subject = ActivityCountFeatureService.build_count_feature(subject_id)
        count_std = np.std(count_feature_subject.iloc[:,1:])
        
        ibi_features_subject = IbiFeatureService.build_hr_features(subject_id)
        ibi_mean = np.mean(ibi_features_subject.iloc[:,1:], axis=0)
        ibi_std = np.std(ibi_features_subject.iloc[:,1:], axis=0)
        
        hr_features_subject = HeartRateFeatureService.build(subject_id, False)
        hr_mean = np.mean(hr_features_subject.iloc[:,1:], axis=0)
        hr_std = np.std(hr_features_subject.iloc[:,1:], axis=0)
        
        normalized_hr_features_subject = HeartRateFeatureService.build(subject_id, True)
        normalized_hr_mean = np.mean(hr_features_subject.iloc[:,1:], axis=0)
        normalized_hr_std = np.std(hr_features_subject.iloc[:,1:], axis=0)
        
        # If there is nothing inside ibi_features_subject, we return
        if not np.any(ibi_features_subject):
            return
        
        
        sleepsessions = BuiltService.get_built_sleepsession_ids(subject_id, FeatureType.cropped, DataSet.usi)
        for session_id in sleepsessions:
    
            if Constants.VERBOSE:
                print("Building USI epoched features for subject " + str(subject_id) + ", session " + str(session_id) + "...")
            
            count_feature = ActivityCountFeatureService.build_count_feature(subject_id, session_id)
            
            # Normalizing count feature, the first row is a timestamp
            count_feature.iloc[:,1:] = count_feature.iloc[:,1:]/count_std
            
            # Normalizing ibi features, the first row is a timestamp
            ibi_features = IbiFeatureService.build_hr_features(subject_id, session_id)
            ibi_features.iloc[:,1:] = (ibi_features.iloc[:,1:] - ibi_mean)/ibi_std      
            
            # If there is nothing inside ibi_features, we continue with the next loop
            if ibi_features.to_numpy().shape()[0] < 2:
                continue
            
            hr_feature = HeartRateFeatureService.build(subject_id, session_id, False)
            hr_feature.iloc[:,1:] = (hr_feature.iloc[:,1:] - hr_mean)/hr_std  
            
            normalized_hr_feature = HeartRateFeatureService.build(subject_id, session_id, True)
            normalized_hr_feature.iloc[:,1:] = (normalized_hr_feature.iloc[:,1:] - normalized_hr_mean)/normalized_hr_std    

            # Create needed folders if they don't already exist
            PathService.create_epoched_folder_path(subject_id, session_id, DataSet.usi)
            
            DataWriter.write_epoched(count_feature, subject_id, session_id, FeatureType.epoched_count, DataSet.usi)
            DataWriter.write_epoched(ibi_features, subject_id, session_id, FeatureType.epoched_ibi, DataSet.usi)
            DataWriter.write_epoched(hr_feature, subject_id, session_id, FeatureType.epoched_hr, DataSet.usi)
            DataWriter.write_epoched(normalized_hr_feature, subject_id, session_id, FeatureType.epoched_normalized_hr, DataSet.usi)
                
        
