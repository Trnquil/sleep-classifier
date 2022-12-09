from source.preprocessing.ibi.ibi_feature_service import IbiFeatureService
from source.preprocessing.activity_count.activity_count_feature_service import ActivityCountFeatureService
from source.preprocessing.raw_data_processor import RawDataProcessor
from source.analysis.setup.feature_type import FeatureType
from source.data_services.data_writer import DataWriter
from source.preprocessing.path_service import PathService
from source.preprocessing.heart_rate.heart_rate_feature_service import HeartRateFeatureService
from source.data_services.dataset import DataSet
from source.data_services.data_loader import DataLoader
from source.exception_logger import ExceptionLogger

import sys

class MssEpochedFeatureBuilder(object):
    
    @staticmethod
    def build(subject_id, session_id):
        try:
            count_feature = ActivityCountFeatureService.build_count_feature(subject_id, session_id, DataSet.mss)
            
            ibi_collection = DataLoader.load_cropped(subject_id, session_id, FeatureType.cropped_ibi, DataSet.mss)
            valid_epochs_ibi = RawDataProcessor.get_valid_epochs([ibi_collection])
            ibi_features = IbiFeatureService.build_from_collection(ibi_collection, valid_epochs_ibi)
            
            # Create needed folders if they don't already exist
            PathService.create_epoched_folder_path(subject_id, session_id, DataSet.mss)
            
            DataWriter.write_epoched(count_feature, subject_id, session_id, FeatureType.epoched_count, DataSet.mss)
            DataWriter.write_epoched(ibi_features, subject_id, session_id, FeatureType.epoched_ibi, DataSet.mss)
        except:
            ExceptionLogger.append_exception(subject_id, session_id, "Epoched", DataSet.usi.name, sys.exc_info()[0])
            print("Error: ", sys.exc_info()[0], " while building MSS epoched features for " + str(subject_id), ", session " + str(session_id))