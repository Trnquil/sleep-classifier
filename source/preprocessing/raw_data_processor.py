import sys
# caution: path[0] is reserved for script path (or '' in REPL)
# We are inserting the sleepclassifier as a module to be able to access source.<xxx>
sys.path.insert(1, '../..')

import numpy as np

from source import utils
from source.preprocessing.activity_count.activity_count_service import ActivityCountService
from source.preprocessing.epoch import Epoch
from source.preprocessing.interval import Interval
from source.analysis.setup.sleep_session_service import SleepSessionService
from source.analysis.setup.feature_type import FeatureType
from source.data_services.data_service import DataService
from source.preprocessing.collection import Collection
from source.data_services.data_loader import DataLoader
from source.data_services.data_writer import DataWriter
from source.preprocessing.collection_service import CollectionService
from multipledispatch import dispatch

class RawDataProcessor:
    BASE_FILE_PATH = utils.get_project_root().joinpath('outputs/cropped/')

    @staticmethod
    def crop_all(subject_id):
        
        '''Loading data'''
        motion_collection = DataLoader.load_raw(subject_id, FeatureType.raw_acc)
        heart_rate_collection = DataLoader.load_raw(subject_id,  FeatureType.raw_hr)
        count_collection = ActivityCountService.build_activity_counts_without_matlab(subject_id, motion_collection.data) # Builds activity counts with python, not MATLAB
        
        
        
        '''Getting intersecting intervals of time for all collections'''
        valid_interval = RawDataProcessor.get_intersecting_interval([motion_collection, heart_rate_collection])


        '''cropping all the data to the valid interval'''
        motion_collection = CollectionService.crop(motion_collection, valid_interval)
        heart_rate_collection = CollectionService.crop(heart_rate_collection, valid_interval)
        x = heart_rate_collection.data
        
        '''splitting each collection into sleepsessions'''
        motion_sleepsession_tuples = SleepSessionService.assign_collection_to_sleepsession(subject_id, motion_collection)
        heart_rate_sleepsession_tuples = SleepSessionService.assign_collection_to_sleepsession(subject_id, heart_rate_collection)
        count_sleepsession_tuples = SleepSessionService.assign_collection_to_sleepsession(subject_id, count_collection)
        
        
        '''writing all the data to disk'''
        for i in range(len(motion_sleepsession_tuples)):
            motion_collection = motion_sleepsession_tuples[i][1]
            heart_rate_collection = heart_rate_sleepsession_tuples[i][1]
            count_collection = count_sleepsession_tuples[i][1]
            
            sleep_session_id = motion_sleepsession_tuples[i][0].session_id
            
            if(np.any(motion_collection.data) and np.any(heart_rate_collection.data) and np.any(count_collection.data)):
                DataWriter.write_cropped(motion_collection, sleep_session_id, FeatureType.cropped_motion)
                DataWriter.write_cropped(heart_rate_collection, sleep_session_id, FeatureType.cropped_heart_rate)
                DataWriter.write_cropped(count_collection, sleep_session_id, FeatureType.cropped_count)
        
                                     
        '''Doing all above steps for normalized data'''
        normalized_heart_rate_collection = Collection(subject_id, DataService.load_feature_raw(subject_id, FeatureType.cropped_heart_rate))
        normalized_heart_rate_collection = RawDataProcessor.normalize(normalized_heart_rate_collection)
        normalized_heart_rate_sleepsession_tuples = SleepSessionService.assign_collection_to_sleepsession(subject_id, normalized_heart_rate_collection)
           

        for normalized_heart_rate_sleepsession_tuple in normalized_heart_rate_sleepsession_tuples:
            normalized_heart_rate_collection = normalized_heart_rate_sleepsession_tuple[1]
            sleep_session_id = normalized_heart_rate_sleepsession_tuple[0].session_id
            
            if(np.any(normalized_heart_rate_collection.data)):
                DataWriter.write_cropped(normalized_heart_rate_collection, sleep_session_id, FeatureType.normalized_heart_rate)
            
    @staticmethod 
    def normalize(collection):
        feature = collection.values
        mean = np.mean(feature, axis=0)
        std = np.std(feature, axis=0)
        
        normalized_feature = (feature - mean)/std
        timestamps = np.expand_dims(collection.timestamps, axis=1)
        normalized_data = np.concatenate((timestamps, normalized_feature), axis=1)
        
        return Collection(subject_id= collection.subject_id, data = normalized_data)
    
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
    @dispatch(str, str)
    def get_valid_epochs(subject_id, session_id):

        motion_collection = DataLoader.load_cropped(subject_id, session_id, FeatureType.cropped_motion)
        heart_rate_collection = DataLoader.load_cropped(subject_id, session_id, FeatureType.cropped_heart_rate)

        #Manually setting the start time to 0
        start_time = 0
        motion_floored_timestamps, motion_epoch_dictionary = RawDataProcessor.get_valid_epoch_dictionary(motion_collection.timestamps,
                                                                              start_time)
        hr_floored_timestamps, hr_epoch_dictionary = RawDataProcessor.get_valid_epoch_dictionary(heart_rate_collection.timestamps,
                                                                          start_time)

        valid_epochs = []
        for timestamp in motion_floored_timestamps:
            index = 0
            if timestamp in motion_epoch_dictionary and timestamp in hr_epoch_dictionary:
                epoch = Epoch(timestamp, index)
                index += 1
                valid_epochs.append(epoch)
        return valid_epochs
    
    @staticmethod
    @dispatch(str)
    def get_valid_epochs(subject_id):

        motion_feature = DataService.load_feature_raw(subject_id, FeatureType.cropped_motion)
        motion_collection = Collection(subject_id, motion_feature)
        
        heart_rate_feature = DataService.load_feature_raw(subject_id, FeatureType.cropped_heart_rate)
        heart_rate_collection = Collection(subject_id, heart_rate_feature)

        #Manually setting the start time to 0
        start_time = 0
        motion_floored_timestamps, motion_epoch_dictionary = RawDataProcessor.get_valid_epoch_dictionary(motion_collection.timestamps,
                                                                              start_time)
        hr_floored_timestamps, hr_epoch_dictionary = RawDataProcessor.get_valid_epoch_dictionary(heart_rate_collection.timestamps,
                                                                          start_time)

        valid_epochs = []
        for timestamp in motion_floored_timestamps:
            index = 0
            if timestamp in motion_epoch_dictionary and timestamp in hr_epoch_dictionary:
                epoch = Epoch(timestamp, index)
                index += 1
                valid_epochs.append(epoch)
        return valid_epochs

    @staticmethod
    def get_valid_epoch_dictionary(timestamps, start_time):
        epoch_dictionary = {}
        floored_timestamps = []

        for ind in range(np.shape(timestamps)[0]):
            time = timestamps[ind]
            
            #This line floors the timespamps to epoch increments
            floored_timestamp = time - np.mod(time - start_time, Epoch.DURATION)

            epoch_dictionary[floored_timestamp] = True
            
            if floored_timestamp not in floored_timestamps:
                floored_timestamps.append(floored_timestamp)

        return floored_timestamps, epoch_dictionary

