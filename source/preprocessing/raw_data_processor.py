import sys
# caution: path[0] is reserved for script path (or '' in REPL)
# We are inserting the sleepclassifier as a module to be able to access source.<xxx>
sys.path.insert(1, '../..')

import numpy as np

from source import utils
from source.preprocessing.activity_count.activity_count_service import ActivityCountService
from source.preprocessing.epoch import Epoch
from source.preprocessing.heart_rate.heart_rate_service import HeartRateService
from source.preprocessing.interval import Interval
from source.preprocessing.motion.motion_service import MotionService
from source.analysis.setup.sleep_session_service import SleepSessionService
from source.analysis.setup.feature_type import FeatureType


class RawDataProcessor:
    BASE_FILE_PATH = utils.get_project_root().joinpath('outputs/cropped/')

    @staticmethod
    def crop_all(subject_id):
        
        motion_collection = MotionService.load_raw(subject_id)
        heart_rate_collection = HeartRateService.load_raw(subject_id)
        
        valid_interval = RawDataProcessor.get_intersecting_interval([motion_collection, heart_rate_collection])

        motion_collection = MotionService.crop(motion_collection, valid_interval)
        heart_rate_collection = HeartRateService.crop(heart_rate_collection, valid_interval)
        
        motion_sleepsession_tuples = SleepSessionService.assign_collection_to_sleepsession(subject_id, motion_collection)
        heart_rate_sleepsession_tuples = SleepSessionService.assign_collection_to_sleepsession(subject_id, heart_rate_collection)
        
        for motion_sleepsession_tuple in motion_sleepsession_tuples:
            motion_collection = motion_sleepsession_tuple[1]
            sleep_session_id = motion_sleepsession_tuple[0].session_id
            
            if(np.any(motion_collection.data)):
                MotionService.write(motion_collection, sleep_session_id)
                ActivityCountService.build_activity_counts_without_matlab(subject_id, motion_collection.data, sleep_session_id)  # Builds activity counts with python, not MATLAB
            
        for heart_rate_sleepsession_tuple in heart_rate_sleepsession_tuples:
            heart_rate_collection = heart_rate_sleepsession_tuple[1]
            sleep_session_id = heart_rate_sleepsession_tuple[0].session_id
            
            if(np.any(heart_rate_collection.data)):
                HeartRateService.write(heart_rate_collection, sleep_session_id)
            
        

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
    def get_valid_epochs(subject_id):

        #psg_collection = PSGService.load_cropped(subject_id)
        motion_collection = MotionService.load_cropped(subject_id)
        heart_rate_collection = HeartRateService.load_cropped(subject_id)

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

