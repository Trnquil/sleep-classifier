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
from source.preprocessing.path_service import PathService
from source.constants import Constants

from multipledispatch import dispatch
import heartpy as hp
from matplotlib import pyplot as plt
import scipy.signal as s
from scipy.signal import butter,filtfilt, iirnotch


class RawDataProcessor:
    BASE_FILE_PATH = utils.get_project_root().joinpath('outputs/cropped/')
    IBI_LOWER_BOUND = 0.4
    IBI_UPPER_BOUND = 2

    @staticmethod
    def crop_all(subject_id):
        
        if Constants.USE_BVP:
            '''Loading data'''
            hr_collection = DataLoader.load_raw(subject_id, FeatureType.raw_hr)
            motion_collection = DataLoader.load_raw(subject_id, FeatureType.raw_acc)
            count_collection = ActivityCountService.build_activity_counts_without_matlab(subject_id, motion_collection.data) # Builds activity counts with python, not MATLAB
            bvp_collection = DataLoader.load_raw(subject_id, FeatureType.raw_bvp)
            ibi_collection = DataLoader.load_raw(subject_id, FeatureType.raw_ibi)
            
            '''Getting intersecting intervals of time for all collections'''
            valid_interval = RawDataProcessor.get_intersecting_interval([motion_collection, bvp_collection])
            
            '''cropping all the data to the valid interval'''
            motion_collection = CollectionService.crop(motion_collection, valid_interval)
            count_collection = CollectionService.crop(count_collection, valid_interval)
            hr_collection = CollectionService.crop(hr_collection, valid_interval)
            bvp_collection = CollectionService.crop(bvp_collection, valid_interval)
            ibi_collection = CollectionService.crop(ibi_collection, valid_interval)

            
            '''splitting each collection into sleepsessions'''
            motion_sleepsession_tuples = SleepSessionService.assign_collection_to_sleepsession(subject_id, motion_collection)
            count_sleepsession_tuples = SleepSessionService.assign_collection_to_sleepsession(subject_id, count_collection)
            hr_sleepsession_tuples = SleepSessionService.assign_collection_to_sleepsession(subject_id, hr_collection)
            bvp_sleepsession_tuples = SleepSessionService.assign_collection_to_sleepsession(subject_id, bvp_collection)
            ibi_sleepsession_tuples = SleepSessionService.assign_collection_to_sleepsession(subject_id, ibi_collection)
            
                    

            '''writing all the data to disk'''
            for i in range(len(motion_sleepsession_tuples)):
                motion_collection = motion_sleepsession_tuples[i][1]
                count_collection = count_sleepsession_tuples[i][1]
                hr_collection = hr_sleepsession_tuples[i][1]
                bvp_collection = bvp_sleepsession_tuples[i][1]
                ibi_collection_usi = ibi_sleepsession_tuples[i][1]
                
                sleep_session_id = motion_sleepsession_tuples[i][0].session_id
                
                if(np.any(motion_collection.data) and np.any(bvp_collection.data) and np.any(count_collection.data)):
                    
                    PathService.create_cropped_file_path(subject_id, sleep_session_id)
                    
                    ibi_collection = RawDataProcessor.get_ibi_from_bvp(bvp_collection)
                    
                    hr_from_bvp = 60/ibi_collection.values
                    hr_from_ibi = 60/ibi_collection_usi.values
                    hr_from_usi = hr_collection.values
                    
                    print("hr mean from bvp: " + str(np.mean(hr_from_bvp)) + ", hr mean from usi: " + str(np.mean(hr_from_usi)) + ", hr mean from ibi: " + str(np.mean(hr_from_ibi)))
                    print("hr std from bvp: " + str(np.std(hr_from_bvp)) + ", hr std from usi: " + str(np.std(hr_from_usi)) + ", hr std from ibi: " + str(np.std(hr_from_ibi)))
                    
                    DataWriter.write_cropped(motion_collection, sleep_session_id, FeatureType.cropped_motion)
                    DataWriter.write_cropped(ibi_collection, sleep_session_id, FeatureType.cropped_ibi)
                    DataWriter.write_cropped(count_collection, sleep_session_id, FeatureType.cropped_count)
                    DataWriter.write_cropped(hr_collection, sleep_session_id, FeatureType.cropped_hr)
                    
        
                    
        elif not Constants.USE_BVP:
            '''Loading data'''
            hr_collection = DataLoader.load_raw(subject_id, FeatureType.raw_hr)
            motion_collection = DataLoader.load_raw(subject_id, FeatureType.raw_acc)
            count_collection = ActivityCountService.build_activity_counts_without_matlab(subject_id, motion_collection.data) # Builds activity counts with python, not MATLAB
            ibi_collection = DataLoader.load_raw(subject_id, FeatureType.raw_ibi)
            
            '''Getting intersecting intervals of time for all collections'''
            valid_interval = RawDataProcessor.get_intersecting_interval([motion_collection, ibi_collection])
            
            '''cropping all the data to the valid interval'''
            motion_collection = CollectionService.crop(motion_collection, valid_interval)
            ibi_collection = CollectionService.crop(ibi_collection, valid_interval)
            hr_collection = CollectionService.crop(hr_collection, valid_interval)
            count_collection = CollectionService.crop(count_collection, valid_interval)
            
            '''splitting each collection into sleepsessions'''
            motion_sleepsession_tuples = SleepSessionService.assign_collection_to_sleepsession(subject_id, motion_collection)
            count_sleepsession_tuples = SleepSessionService.assign_collection_to_sleepsession(subject_id, count_collection)
            hr_sleepsession_tuples = SleepSessionService.assign_collection_to_sleepsession(subject_id, hr_collection)
            ibi_sleepsession_tuples = SleepSessionService.assign_collection_to_sleepsession(subject_id, ibi_collection)
            
                    

            '''writing all the data to disk'''
            for i in range(len(motion_sleepsession_tuples)):
                motion_collection = motion_sleepsession_tuples[i][1]
                count_collection = count_sleepsession_tuples[i][1]
                hr_collection = hr_sleepsession_tuples[i][1]
                ibi_collection = ibi_sleepsession_tuples[i][1]
                
                sleep_session_id = motion_sleepsession_tuples[i][0].session_id
                
                if(np.any(motion_collection.data) and np.any(ibi_collection.data) and np.any(count_collection.data)):
                    
                    PathService.create_cropped_file_path(subject_id, sleep_session_id)
                    
                    DataWriter.write_cropped(motion_collection, sleep_session_id, FeatureType.cropped_motion)
                    DataWriter.write_cropped(ibi_collection, sleep_session_id, FeatureType.cropped_ibi)
                    DataWriter.write_cropped(count_collection, sleep_session_id, FeatureType.cropped_count)
                    DataWriter.write_cropped(hr_collection, sleep_session_id, FeatureType.cropped_hr)
            

            
                                        
    
    def get_ibi_from_bvp(bvp_collection):
        # Working on BVP values to produce IBI sequence
        bvp_values = bvp_collection.values.squeeze()
        filtered = RawDataProcessor.bvp_filter(bvp_values)
        working_data, measures = hp.process(filtered, bvp_collection.data_frequency)
        ibi_values = working_data['RR_list']/1000
        timestamps_ibi = np.cumsum(ibi_values) + bvp_collection.timestamps[0]
        data = np.stack((timestamps_ibi, ibi_values), axis=1)
        
        # Making sure that the timestamp drift between derived ibi and original BVP timestamps is not too large
        assert bvp_collection.timestamps[-1] - timestamps_ibi[-1] < 10, "Ibi drift over 10 seconds"
        
        #only keeping IBI values between 0.4 and 2
        mask = [RawDataProcessor.IBI_LOWER_BOUND < x < RawDataProcessor.IBI_UPPER_BOUND for x in ibi_values]
        data = data[mask]
        ibi_collection = Collection(bvp_collection.subject_id, data, 0)
        return ibi_collection
    
    
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
    def bvp_filter(signal):
        fH = 4
        fL = 0.5
        freq = 64
        nyquist = freq / 2
        sos = s.cheby2(4, 20, Wn=(fL / nyquist, fH / nyquist), btype='bandpass', output = 'sos')
        filtered = s.sosfiltfilt(sos, signal)
        return filtered

    
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
        ibi_collection = DataLoader.load_cropped(subject_id, session_id, FeatureType.cropped_ibi)

        #Manually setting the start time to 0
        start_time = 0
        motion_floored_timestamps, motion_epoch_dictionary = RawDataProcessor.get_valid_epoch_dictionary(motion_collection.timestamps,
                                                                              start_time)
        ibi_floored_timestamps, ibi_epoch_dictionary = RawDataProcessor.get_valid_epoch_dictionary(ibi_collection.timestamps,
                                                                          start_time)

        valid_epochs = []
        for timestamp in motion_floored_timestamps:
            index = 0
            if timestamp in motion_epoch_dictionary and timestamp in ibi_epoch_dictionary:
                epoch = Epoch(timestamp, index)
                index += 1
                valid_epochs.append(epoch)
        return valid_epochs
    
    @staticmethod
    @dispatch(str)
    def get_valid_epochs(subject_id):

        motion_feature = DataService.load_feature_raw(subject_id, FeatureType.cropped_motion)
        motion_collection = Collection(subject_id, motion_feature, 0)
        
        ibi_feature = DataService.load_feature_raw(subject_id, FeatureType.cropped_ibi)
        ibi_collection = Collection(subject_id, ibi_feature, 0)

        #Manually setting the start time to 0
        start_time = 0
        motion_floored_timestamps, motion_epoch_dictionary = RawDataProcessor.get_valid_epoch_dictionary(motion_collection.timestamps,
                                                                              start_time)
        ibi_floored_timestamps, ibi_epoch_dictionary = RawDataProcessor.get_valid_epoch_dictionary(ibi_collection.timestamps,
                                                                          start_time)

        valid_epochs = []
        for timestamp in motion_floored_timestamps:
            index = 0
            if timestamp in motion_epoch_dictionary and timestamp in ibi_epoch_dictionary:
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

