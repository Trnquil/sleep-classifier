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
from source.preprocessing.sleep_wake import SleepWake

import sys

class MssEpochedFeatureBuilder(object):
    
    @staticmethod
    def build(subject_id, session_id):
        try:
            count_feature = ActivityCountFeatureService.build_count_feature(subject_id, session_id, SleepWake.sleep, DataSet.mss)
            
            hr_collection = DataLoader.load_cropped(subject_id, session_id, FeatureType.cropped_hr, SleepWake.sleep, DataSet.mss)
            valid_epochs_hr = UsiRawDataProcessor.get_valid_epochs([hr_collection])
            hr_features = HeartRateFeatureService.build_from_collection(hr_collection, valid_epochs_hr)
            
            hrv_features = MssLoader.load_epoched_hrv_features(subject_id, session_id)
            hrv_features = hrv_features
            prefixed_columns = ['mss_ibi_' + str(val) if val!="epoch_timestamp" else str(val) for val in hrv_features.columns]
            hrv_features.columns = prefixed_columns
            
            DataWriter.write_epoched(hrv_features, subject_id, session_id, FeatureType.epoched_ibi_mss, SleepWake.sleep, DataSet.mss)
            DataWriter.write_epoched(count_feature, subject_id, session_id, FeatureType.epoched_count, SleepWake.sleep, DataSet.mss)
            DataWriter.write_epoched(hr_features, subject_id, session_id, FeatureType.epoched_hr, SleepWake.sleep, DataSet.mss)
        except:
            ExceptionLogger.append_exception(subject_id, session_id, "Epoched", DataSet.usi.name, sys.exc_info()[0])
            print("Skip subject ", str(subject_id), ", session ", str(session_id), " due to ", sys.exc_info()[0])
