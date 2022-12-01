from source.mesa.mesa_heart_rate_service import MesaHeartRateService
from source.preprocessing.heart_rate.heart_rate_feature_service import HeartRateFeatureService
from source.mesa.mesa_actigraphy_service import MesaActigraphyService
from source.mesa.mesa_psg_service import MesaPSGService
from source.preprocessing.raw_data_processor import RawDataProcessor
from source.preprocessing.path_service import PathService
from source.data_services.dataset import DataSet
from source.data_services.data_writer import DataWriter
from source.analysis.setup.feature_type import FeatureType
from source.constants import Constants
from source.preprocessing.interval import Interval
from source.preprocessing.feature_service import FeatureService
from source.preprocessing.epoch import Epoch
from source.preprocessing.activity_count.activity_count_feature_service import ActivityCountFeatureService
from source.preprocessing.collection_service import CollectionService
from source.preprocessing.epoched_feature_builder import EpochedFeatureBuilder
from source.exception_logger import ExceptionLogger

import numpy as np
import pandas as pd
import sys

class MesaFeatureBuilder(object):
    
    @staticmethod
    def build(subject_id):
            
        try:
            raw_labeled_sleep = MesaPSGService.load_raw(subject_id)
            heart_rate_collection = MesaHeartRateService.load_raw(subject_id)
            activity_count_collection = MesaActigraphyService.load_raw(subject_id)
    
            if activity_count_collection.data[0][0] != -1:
    
                interval = Interval(start_time=0, end_time=np.shape(raw_labeled_sleep)[0])
    
                activity_count_collection = CollectionService.crop(activity_count_collection, interval)
                heart_rate_collection = CollectionService.crop(heart_rate_collection, interval)
                
                valid_epochs = []
    
                for timestamp in range(interval.start_time, interval.end_time, Epoch.DURATION):
                    epoch = Epoch(timestamp=timestamp, index=len(valid_epochs))
                    activity_count_indices = FeatureService.get_window(activity_count_collection.timestamps,
                                                                                    epoch)
                    heart_rate_indices = FeatureService.get_window(heart_rate_collection.timestamps, epoch)
    
                    if len(activity_count_indices) > 0 and 0 not in heart_rate_collection.values[heart_rate_indices]:
                        valid_epochs.append(epoch)
                    else:
                        pass
    
                labeled_sleep = MesaPSGService.crop(psg_labels=raw_labeled_sleep, valid_epochs=valid_epochs)
                labeled_sleep = pd.DataFrame(labeled_sleep)
                labeled_sleep.columns = ['epoch_timestamp', 'sleep_label']
    
                count_feature = ActivityCountFeatureService.build_from_collection(activity_count_collection,
                                                                                  valid_epochs)
                
                hr_features = HeartRateFeatureService.build_from_collection(heart_rate_collection, valid_epochs)
                
    
                # Writing features to disk
    
                # Create needed folders if they don't already exist
                PathService.create_epoched_folder_path(subject_id, 'SS_01', DataSet.mesa)
                DataWriter.write_epoched(hr_features, subject_id, 'SS_01', FeatureType.epoched_hr, DataSet.mesa)
                DataWriter.write_epoched(count_feature, subject_id, 'SS_01', FeatureType.epoched_count, DataSet.mesa)
                DataWriter.write_epoched(labeled_sleep, subject_id, 'SS_01', FeatureType.epoched_sleep_label, DataSet.mesa)
        except:
            ExceptionLogger.append_exception(subject_id, "SS_01", "Epoched", DataSet.mesa.name)
            print("Error: ", sys.exc_info()[0], " while building MESA feature for subject " + str(subject_id))

            