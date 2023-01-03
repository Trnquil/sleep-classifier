import sys

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
from source.runner_parameters import RunnerParameters
from source.data_services.mss_loader import MssLoader
from source.preprocessing.sleep_wake import SleepWake

from GEMINI.gemini_service import GeminiService


class UsiEpochedFeatureBuilder(object):

    @staticmethod
    def build(subject_id, session_id, sleep_wake):
               
        try:
            count_feature = ActivityCountFeatureService.build_count_feature(subject_id, session_id, sleep_wake, DataSet.usi)
            
            heart_rate_collection = DataLoader.load_cropped(subject_id, session_id, FeatureType.cropped_hr, sleep_wake, DataSet.usi)
            valid_epochs_hr = UsiRawDataProcessor.get_valid_epochs([heart_rate_collection])
            hr_feature = HeartRateFeatureService.build_from_collection(heart_rate_collection, valid_epochs_hr)
            
            if sleep_wake.name == SleepWake.selfreported_sleep.name or sleep_wake.name == SleepWake.sleep.name:
                model = GeminiService.load_model()
                hr_gemini_clusters = HeartRateFeatureService.build_GEMINI_clusters(heart_rate_collection, valid_epochs_hr, model)
            
            ibi_collection = DataLoader.load_cropped(subject_id, session_id, FeatureType.cropped_ibi, sleep_wake, DataSet.usi)
            
            valid_epochs_ibi = UsiRawDataProcessor.get_valid_epochs([ibi_collection])
            ibi_features = IbiFeatureService.build_from_collection(ibi_collection, DataSet.usi, valid_epochs_ibi)
            
            ibi_collection_from_ppg = DataLoader.load_cropped(subject_id, session_id, FeatureType.cropped_ibi_from_ppg, sleep_wake, DataSet.usi)
            valid_epochs_ibi_from_ppg = UsiRawDataProcessor.get_valid_epochs([ibi_collection_from_ppg])
            ibi_features_from_ppg = IbiFeatureService.build_from_collection(ibi_collection_from_ppg, DataSet.usi, valid_epochs_ibi_from_ppg)
            
            ibi_mss_values = ['ibi_' + str(val) if val!="epoch_timestamp" else str(val) for val in MssLoader.MSS_feature_dict.values()]
            ibi_mss_features = ibi_features_from_ppg[ibi_mss_values]
            prefixed_columns = ['mss_' + str(val) if val!="epoch_timestamp" else str(val) for val in ibi_mss_features.columns]
            ibi_mss_features.columns = prefixed_columns

            DataWriter.write_epoched(ibi_mss_features, subject_id, session_id, FeatureType.epoched_ibi_mss, sleep_wake, DataSet.usi)
            DataWriter.write_epoched(count_feature, subject_id, session_id, FeatureType.epoched_count, sleep_wake, DataSet.usi)
            DataWriter.write_epoched(hr_feature, subject_id, session_id, FeatureType.epoched_hr, sleep_wake, DataSet.usi)
            DataWriter.write_epoched(ibi_features, subject_id, session_id, FeatureType.epoched_ibi, sleep_wake, DataSet.usi)
            DataWriter.write_epoched(ibi_features_from_ppg, subject_id, session_id, FeatureType.epoched_ibi_from_ppg, sleep_wake, DataSet.usi)
            
            if sleep_wake.name == SleepWake.selfreported_sleep.name or sleep_wake.name == SleepWake.sleep.name:
                DataWriter.write_epoched(hr_gemini_clusters, subject_id, session_id, FeatureType.epoched_cluster_GEMINI, sleep_wake, DataSet.usi)

        except:
            ExceptionLogger.append_exception(subject_id, session_id, "Epoched", DataSet.usi.name, sys.exc_info()[0])
            print("Skip subject ", str(subject_id), ", session ", str(session_id), " due to ", sys.exc_info()[0])
    
        
