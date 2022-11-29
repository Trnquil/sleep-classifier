import sys
sys.path.insert(1, '../..')

import numpy as np

from source import utils
from source.preprocessing.activity_count.activity_count_service import ActivityCountService
from source.preprocessing.epoch import Epoch
from source.preprocessing.interval import Interval
from source.analysis.setup.sleep_session_service import SleepSessionService
from source.analysis.setup.feature_type import FeatureType
from source.preprocessing.collection import Collection
from source.data_services.data_loader import DataLoader
from source.data_services.data_writer import DataWriter
from source.preprocessing.collection_service import CollectionService
from source.preprocessing.path_service import PathService
from source.preprocessing.bvp_service import BvpService
from source.constants import Constants


from multipledispatch import dispatch


class RawDataProcessor:
    BASE_FILE_PATH = utils.get_project_root().joinpath('outputs/cropped/')
    
    @staticmethod
    def crop_all(subject_id):
        

        '''Loading data'''
        hr_collection = DataLoader.load_raw(subject_id, FeatureType.raw_hr)
        motion_collection = DataLoader.load_raw(subject_id, FeatureType.raw_acc)
        count_collection = ActivityCountService.build_activity_counts_without_matlab(subject_id, motion_collection.data) # Builds activity counts with python, not MATLAB
        bvp_collection = DataLoader.load_raw(subject_id, FeatureType.raw_bvp)
        ibi_collection = DataLoader.load_raw(subject_id, FeatureType.raw_ibi)
        normalized_hr_collection = RawDataProcessor.normalize(hr_collection)

        
        '''splitting each collection into sleepsessions'''
        motion_sleepsession_tuples = SleepSessionService.assign_collection_to_sleepsession(subject_id, motion_collection)
        count_sleepsession_tuples = SleepSessionService.assign_collection_to_sleepsession(subject_id, count_collection)
        hr_sleepsession_tuples = SleepSessionService.assign_collection_to_sleepsession(subject_id, hr_collection)
        bvp_sleepsession_tuples = SleepSessionService.assign_collection_to_sleepsession(subject_id, bvp_collection)
        ibi_sleepsession_tuples = SleepSessionService.assign_collection_to_sleepsession(subject_id, ibi_collection)
        normalized_hr_sleepsession_tuples = SleepSessionService.assign_collection_to_sleepsession(subject_id, normalized_hr_collection)
        
                

        '''writing all the data to disk'''
        for i in range(len(motion_sleepsession_tuples)):
            try:
                motion_collection = motion_sleepsession_tuples[i][1]
                count_collection = count_sleepsession_tuples[i][1]
                hr_collection = hr_sleepsession_tuples[i][1]
                bvp_collection = bvp_sleepsession_tuples[i][1]
                ibi_collection = ibi_sleepsession_tuples[i][1]
                normalized_hr_collection = normalized_hr_sleepsession_tuples[i][1]
                
                ibi_collection_from_pgg = BvpService.get_ibi_from_bvp(bvp_collection)
                
                session_id = motion_sleepsession_tuples[i][0].session_id
                
                if Constants.VERBOSE:
                    print("Writing cropped data from subject " + str(subject_id) +  ", session " + str(session_id) + "...")
                
                PathService.create_cropped_file_path(subject_id, session_id)
                
                DataWriter.write_cropped(motion_collection, session_id, FeatureType.cropped_motion)
                DataWriter.write_cropped(ibi_collection, session_id, FeatureType.cropped_ibi)
                DataWriter.write_cropped(ibi_collection_from_pgg, session_id, FeatureType.cropped_ibi_from_ppg)
                DataWriter.write_cropped(count_collection, session_id, FeatureType.cropped_count)
                DataWriter.write_cropped(hr_collection, session_id, FeatureType.cropped_hr)
                DataWriter.write_cropped(normalized_hr_collection, session_id, FeatureType.normalized_hr)
            except:
                print("Error: ", sys.exc_info()[0], " while building cropped features for " + str(subject_id), ", session " + str(session_id))
    
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
            timestamps, dictionary = RawDataProcessor.get_valid_epoch_dictionary(collection.timestamps)
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
        start_time = 0
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

