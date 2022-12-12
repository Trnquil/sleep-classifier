import sys
sys.path.insert(1, '../..')

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



class UsiEpochedFeatureBuilder(object):

    @staticmethod
    def build(subject_id, session_id):    
               
        try:
            count_feature = ActivityCountFeatureService.build_count_feature(subject_id, session_id, DataSet.usi)
            
            hr_feature = HeartRateFeatureService.build(subject_id, session_id, DataSet.usi)
            
            ibi_collection = DataLoader.load_cropped(subject_id, session_id, FeatureType.cropped_ibi, DataSet.usi)
            valid_epochs_ibi = RawDataProcessor.get_valid_epochs([ibi_collection])
            ibi_features = IbiFeatureService.build_from_collection(ibi_collection, DataSet.usi, valid_epochs_ibi) 
            
            ibi_collection_from_ppg = DataLoader.load_cropped(subject_id, session_id, FeatureType.cropped_ibi_from_ppg, DataSet.usi)
            valid_epochs_ibi_from_ppg = RawDataProcessor.get_valid_epochs([ibi_collection_from_ppg])
            ibi_features_from_ppg = IbiFeatureService.build_from_collection(ibi_collection_from_ppg, DataSet.usi, valid_epochs_ibi_from_ppg)    

            
            DataWriter.write_epoched(count_feature, subject_id, session_id, FeatureType.epoched_count, DataSet.usi)
            DataWriter.write_epoched(hr_feature, subject_id, session_id, FeatureType.epoched_hr, DataSet.usi)
            DataWriter.write_epoched(ibi_features, subject_id, session_id, FeatureType.epoched_ibi, DataSet.usi)
            DataWriter.write_epoched(ibi_features_from_ppg, subject_id, session_id, FeatureType.epoched_ibi_from_ppg, DataSet.usi)
        except:
            ExceptionLogger.append_exception(subject_id, session_id, "Epoched", DataSet.usi.name, sys.exc_info()[0])
            print("Skip subject ", str(subject_id), ", session ", str(session_id), " due to ", sys.exc_info()[0])
        
