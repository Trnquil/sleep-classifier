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
from source.data_services.data_loader import DataLoader
from source.exception_logger import ExceptionLogger

import numpy as np
import pandas as pd
from multipledispatch import dispatch
import warnings


class EpochedFeatureBuilder(object):

    @staticmethod
    def build(subject_id, session_id):    
               
        try:
            count_feature = ActivityCountFeatureService.build_count_feature(subject_id, session_id)
            
            hr_feature = HeartRateFeatureService.build(subject_id, session_id)
            
            ibi_collection = DataLoader.load_cropped(subject_id, session_id, FeatureType.cropped_ibi)
            valid_epochs_ibi = RawDataProcessor.get_valid_epochs([ibi_collection])
            ibi_features = IbiFeatureService.build_from_collection(ibi_collection, valid_epochs_ibi) 
            
            ibi_collection_from_ppg = DataLoader.load_cropped(subject_id, session_id, FeatureType.cropped_ibi_from_ppg)
            valid_epochs_ibi_from_ppg = RawDataProcessor.get_valid_epochs([ibi_collection_from_ppg])
            ibi_features_from_ppg = IbiFeatureService.build_from_collection(ibi_collection_from_ppg, valid_epochs_ibi_from_ppg)    
            
            # Create needed folders if they don't already exist
            PathService.create_epoched_folder_path(subject_id, session_id, DataSet.usi)
            
            DataWriter.write_epoched(count_feature, subject_id, session_id, FeatureType.epoched_count, DataSet.usi)
            DataWriter.write_epoched(hr_feature, subject_id, session_id, FeatureType.epoched_hr, DataSet.usi)
            DataWriter.write_epoched(ibi_features, subject_id, session_id, FeatureType.epoched_ibi, DataSet.usi)
            DataWriter.write_epoched(ibi_features_from_ppg, subject_id, session_id, FeatureType.epoched_ibi_from_ppg, DataSet.usi)
        except:
            ExceptionLogger.append_exception(subject_id, session_id, "Epoched", DataSet.usi.name, sys.exc_info()[0])
            print("Error: ", sys.exc_info()[0], " while building epoched features for " + str(subject_id), ", session " + str(session_id))
        
