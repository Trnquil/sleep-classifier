from source.mesa.mesa_heart_rate_service import MesaHeartRateService
from source.preprocessing.heart_rate.heart_rate_feature_service import HeartRateFeatureService
from source.mesa.mesa_actigraphy_service import MesaActigraphyService
from source.mesa.mesa_psg_service import MesaPSGService
from source.preprocessing.usi_raw_data_processor import UsiRawDataProcessor
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
from source.preprocessing.usi_epoched_feature_builder import UsiEpochedFeatureBuilder
from source.exception_logger import ExceptionLogger
from source.mesa.mesa_ppg_service import MesaPPGService
from source.preprocessing.ibi.ibi_feature_service import IbiFeatureService
from source.preprocessing.bvp_service import BvpService
from source.preprocessing.collection import Collection
from source.data_services.mss_loader import MssLoader
from source.preprocessing.sleep_wake import SleepWake

import numpy as np
import pandas as pd
import sys

class MesaFeatureBuilder(object):
    
    @staticmethod
    def build(subject_id):
            
        # try:
            raw_labeled_sleep = MesaPSGService.load_raw(subject_id)
            heart_rate_collection = MesaHeartRateService.load_raw(subject_id)
            activity_count_collection = MesaActigraphyService.load_raw(subject_id)
            bvp_collection = MesaPPGService.load_raw(subject_id)
            
            bvp_downsampled_collection = MesaFeatureBuilder.downsample_signal(bvp_collection, 4)
            
            ibi_collection = BvpService.get_ibi_from_bvp_segmentwise(bvp_downsampled_collection)
            
            
            if activity_count_collection.data[0][0] != -1:
    
                interval = Interval(start_time=0, end_time=np.shape(raw_labeled_sleep)[0])
    
                activity_count_collection = CollectionService.crop(activity_count_collection, interval)
                heart_rate_collection = CollectionService.crop(heart_rate_collection, interval)
                ibi_collection = CollectionService.crop(ibi_collection, interval)
                
                valid_epochs = []
    
                for timestamp in range(interval.start_time, interval.end_time, Epoch.DURATION):
                    epoch = Epoch(timestamp=timestamp, index=len(valid_epochs))
                    activity_count_indices = FeatureService.get_window(activity_count_collection.timestamps,
                                                                                    epoch)
                    heart_rate_indices = FeatureService.get_window(heart_rate_collection.timestamps, epoch)
                    ibi_indices = FeatureService.get_window(ibi_collection.timestamps, epoch)
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
                

                ibi_features_from_ppg = IbiFeatureService.build_from_collection(ibi_collection, DataSet.mesa, valid_epochs)
                
                ibi_mss_values = {'ibi_' + str(val) if val!="epoch_timestamp" else str(val) for val in MssLoader.MSS_feature_dict.values()}
                ibi_mss_features = ibi_features_from_ppg[list(ibi_mss_values)]
                prefixed_columns = ['mss_' + str(val) if val!="epoch_timestamp" else str(val) for val in ibi_mss_features.columns]
                ibi_mss_features.columns = prefixed_columns
                
    
                # Writing features to disk
                DataWriter.write_epoched(ibi_mss_features, subject_id, 'SS_01', FeatureType.epoched_ibi_mss, SleepWake.sleep, DataSet.mesa)
                DataWriter.write_epoched(hr_features, subject_id, 'SS_01', FeatureType.epoched_hr, SleepWake.sleep, DataSet.mesa)
                DataWriter.write_epoched(ibi_features_from_ppg, subject_id, 'SS_01', FeatureType.epoched_ibi_from_ppg, SleepWake.sleep, DataSet.mesa)
                DataWriter.write_epoched(count_feature, subject_id, 'SS_01', FeatureType.epoched_count, SleepWake.sleep, DataSet.mesa)
                DataWriter.write_epoched(labeled_sleep, subject_id, 'SS_01', FeatureType.epoched_sleep_label, SleepWake.sleep, DataSet.mesa)
        # except:
        #     ExceptionLogger.append_exception(subject_id, "SS_01", "Epoched", DataSet.mesa.name, sys.exc_info()[0])
        #     print("Skip subject ", str(subject_id), " due to ", sys.exc_info()[0])
    
    @staticmethod
    def downsample_signal(signal_collection, factor):
        data = signal_collection.data
        data_frequency = signal_collection.data_frequency
        
        subsampled_data = data[0::factor, :]
        subsampled_frequency = data_frequency/factor
        
        return Collection(signal_collection.subject_id, subsampled_data, subsampled_frequency)

            