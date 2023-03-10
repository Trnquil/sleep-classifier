import sys
import numpy as np

from source import utils
from source.preprocessing.activity_count.activity_count_service import ActivityCountService
from source.preprocessing.epoch import Epoch
from source.preprocessing.interval import Interval
from source.preprocessing.sleep_session_services.sleep_session_service import SleepSessionService
from source.analysis.setup.feature_type import FeatureType
from source.preprocessing.collection import Collection
from source.data_services.data_loader import DataLoader
from source.data_services.data_writer import DataWriter
from source.preprocessing.collection_service import CollectionService
from source.preprocessing.path_service import PathService
from source.preprocessing.bvp_service import BvpService
from source.constants import Constants
from source.data_services.dataset import DataSet
from source.runner_parameters import RunnerParameters
from source.exception_logger import ExceptionLogger
from source.preprocessing.sleep_session_services.usi_sleep_session_service import UsiSleepSessionService
from source.preprocessing.sleep_wake import SleepWake

from multipledispatch import dispatch


class UsiRawDataProcessor:
    BASE_FILE_PATH = utils.get_project_root().joinpath('outputs/cropped/')
    
    @staticmethod
    def crop_all(subject_id, sleep_wake):
        

        '''Loading data'''
        motion_collection = DataLoader.load_raw(subject_id, FeatureType.raw_acc)
        hr_collection = DataLoader.load_raw(subject_id, FeatureType.raw_hr)
        bvp_collection = DataLoader.load_raw(subject_id, FeatureType.raw_bvp)
        ibi_collection = DataLoader.load_raw(subject_id, FeatureType.raw_ibi)
        normalized_hr_collection = UsiRawDataProcessor.normalize(hr_collection)
        
        
        '''splitting each collection into sleepsessions'''
        if sleep_wake.name == sleep_wake.selfreported_sleep.name:
            sleepsessions = UsiSleepSessionService.load_selfreported_sleep(subject_id)
        elif sleep_wake.name == sleep_wake.sleep.name:
            sleepsessions = UsiSleepSessionService.load_sleep(subject_id)
        if sleep_wake.name == sleep_wake.wake.name:
            sleepsessions = UsiSleepSessionService.load_wake(subject_id)
        
        motion_sleepsession_tuples = SleepSessionService.assign_collection_to_sleepsession(motion_collection, sleepsessions)
        hr_sleepsession_tuples = SleepSessionService.assign_collection_to_sleepsession(hr_collection, sleepsessions)
        bvp_sleepsession_tuples = SleepSessionService.assign_collection_to_sleepsession(bvp_collection, sleepsessions)
        ibi_sleepsession_tuples = SleepSessionService.assign_collection_to_sleepsession(ibi_collection, sleepsessions)
        normalized_hr_sleepsession_tuples = SleepSessionService.assign_collection_to_sleepsession(normalized_hr_collection, sleepsessions)
        
                

        '''writing all the data to disk'''
        for i in range(len(motion_sleepsession_tuples)):
            try:
                session_id = motion_sleepsession_tuples[i][0].session_id
                
                motion_collection = motion_sleepsession_tuples[i][1]
                hr_collection = hr_sleepsession_tuples[i][1]
                bvp_collection = bvp_sleepsession_tuples[i][1]
                ibi_collection = ibi_sleepsession_tuples[i][1]
                normalized_hr_collection = normalized_hr_sleepsession_tuples[i][1]
                                
                if(np.any(motion_collection.data)):
                    count_collection = ActivityCountService.build_activity_counts_without_matlab(subject_id, motion_collection.data)
                    DataWriter.write_cropped(count_collection, session_id, FeatureType.cropped_count, sleep_wake, DataSet.usi)
                    DataWriter.write_cropped(motion_collection, session_id, FeatureType.cropped_motion, sleep_wake, DataSet.usi)
                    
                if(np.any(ibi_collection.data)):
                    DataWriter.write_cropped(ibi_collection, session_id, FeatureType.cropped_ibi, sleep_wake, DataSet.usi)
                
                if(np.any(bvp_collection.data)):
                    if RunnerParameters.PROCESS_USI_BVP_SEGMENTWISE or sleep_wake.name != SleepWake.selfreported_sleep.name:
                        ibi_collection_from_ppg = BvpService.get_ibi_from_bvp_segmentwise(bvp_collection)
                    else:
                        ibi_collection_from_ppg = BvpService.get_ibi_from_bvp(bvp_collection)
                        
                    DataWriter.write_cropped(ibi_collection_from_ppg, session_id, FeatureType.cropped_ibi_from_ppg, sleep_wake, DataSet.usi)
                    
                if(np.any(hr_collection.data)):
                    DataWriter.write_cropped(hr_collection, session_id, FeatureType.cropped_hr, sleep_wake, DataSet.usi)
                    
                if(np.any(normalized_hr_collection.data)):
                    DataWriter.write_cropped(normalized_hr_collection, session_id, FeatureType.cropped_normalized_hr, sleep_wake, DataSet.usi)
                    
            except:
                ExceptionLogger.append_exception(subject_id, session_id, "Cropped", DataSet.usi.name, sys.exc_info()[0])
                print("Skip subject ", str(subject_id), ", session ", str(session_id), " due to ", sys.exc_info()[0])
    
    @staticmethod 
    def normalize(collection):
        feature = collection.values
        mean = np.mean(feature, axis=0)
        std = np.std(feature, axis=0)
        
        normalized_feature = (feature - mean)/std
        timestamps = np.expand_dims(collection.timestamps, axis=1)
        normalized_data = np.concatenate((timestamps, normalized_feature), axis=1)
        
        return Collection(subject_id= collection.subject_id, data = normalized_data, data_frequency=collection.data_frequency)
    
    @staticmethod
    def get_intersecting_interval(collection_list):
        start_times = []
        end_times = []
        for collection in collection_list:
            interval = collection.get_interval()
            start_times.append(interval.start_time)
            end_times.append(interval.end_time)

        return Interval(start_time=max(start_times), end_time=min(end_times))
    
    @staticmethod
    @dispatch(list)
    def get_valid_epochs(collections):
        
        timestamps_array = []
        dictionary_array = []
        for collection in collections:
            timestamps, dictionary = UsiRawDataProcessor.get_valid_epoch_dictionary(collection.timestamps)
            timestamps_array.append(timestamps)
            dictionary_array.append(dictionary)

        valid_epochs = []
        for timestamp in timestamps_array[0]:
            index = 0
            in_dictionary = [timestamp in dictionary for dictionary in dictionary_array]
            # If timestamp is in all dictionaries, we add the epoch to the valid epochs list
            if (all(item is True for item in in_dictionary)):
                epoch = Epoch(timestamp, index)
                index += 1
                valid_epochs.append(epoch)
        return valid_epochs
        
    @staticmethod
    def get_valid_epoch_dictionary(timestamps):
        epoch_dictionary = {}
        floored_timestamps = []

        for ind in range(np.shape(timestamps)[0]):
            time = timestamps[ind]
            
            #This line floors the timespamps to epoch increments
            floored_timestamp = time - np.mod(time, Epoch.DURATION)

            epoch_dictionary[floored_timestamp] = True
            
            if floored_timestamp not in floored_timestamps:
                floored_timestamps.append(floored_timestamp)

        return floored_timestamps, epoch_dictionary

