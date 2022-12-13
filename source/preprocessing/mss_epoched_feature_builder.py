from source.preprocessing.ibi.ibi_feature_service import IbiFeatureService
from source.preprocessing.activity_count.activity_count_feature_service import ActivityCountFeatureService
from source.preprocessing.usi_raw_data_processor import UsiRawDataProcessor
from source.analysis.setup.feature_type import FeatureType
from source.data_services.data_writer import DataWriter
from source.preprocessing.path_service import PathService
from source.preprocessing.heart_rate.heart_rate_feature_service import HeartRateFeatureService
from source.data_services.dataset import DataSet
from source.data_services.data_loader import DataLoader
from source.exception_logger import ExceptionLogger
from source.data_services.mss_loader import MssLoader

import sys

class MssEpochedFeatureBuilder(object):
    
    @staticmethod
    def build(subject_id, session_id):
        try:
            count_feature = ActivityCountFeatureService.build_count_feature(subject_id, session_id, DataSet.mss)
            
            ibi_collection = DataLoader.load_cropped(subject_id, session_id, FeatureType.cropped_ibi, DataSet.mss)
            valid_epochs_ibi = UsiRawDataProcessor.get_valid_epochs([ibi_collection])
            ibi_features = IbiFeatureService.build_from_collection(ibi_collection, DataSet.mss, valid_epochs_ibi)
            
            DataWriter.write_epoched(count_feature, subject_id, session_id, FeatureType.epoched_count, DataSet.mss)
            DataWriter.write_epoched(ibi_features, subject_id, session_id, FeatureType.epoched_ibi, DataSet.mss)
        except:
            ExceptionLogger.append_exception(subject_id, session_id, "Epoched", DataSet.usi.name, sys.exc_info()[0])
            print("Skip subject ", str(subject_id), ", session ", str(session_id), " due to ", sys.exc_info()[0])
