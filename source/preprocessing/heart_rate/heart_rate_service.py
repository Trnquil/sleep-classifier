import sys
# caution: path[0] is reserved for script path (or '' in REPL)
# We are inserting the sleepclassifier as a module 
sys.path.insert(1, '../../..')


import numpy as np
import pandas as pd
import os

from source import utils
from source.constants import Constants
from source.preprocessing.heart_rate.heart_rate_collection import HeartRateCollection


class HeartRateService(object):

    @staticmethod
    def load_raw(subject_id):
        raw_hr_path = HeartRateService.get_raw_file_path(subject_id)
        
        #delimiter doesn't do anything but I don't want to break the rest of the program
        heart_rate_array = HeartRateService.load(raw_hr_path, ",")
        heart_rate_array = utils.remove_repeats(heart_rate_array)
        return HeartRateCollection(subject_id=subject_id, data=heart_rate_array)

    @staticmethod
    def load_cropped(subject_id):
        cropped_hr_path = HeartRateService.get_cropped_file_path(subject_id)
        heart_rate_array = pd.read_csv(str(cropped_hr_path), delimiter=' ', header=None)
        heart_rate_array = heart_rate_array.values
        return HeartRateCollection(subject_id=subject_id, data=heart_rate_array)

    #delimiter doesn't do anything but I don't want to break the rest of the program
    @staticmethod
    def load(hr_file, delimiter=" "):
        heart_rate_array = pd.read_csv(str(hr_file), delimiter=delimiter)
        
        #unix start time of the data
        start_time = heart_rate_array.columns.values.tolist()
        start_time = float(start_time[0])
        
        #Let us center the time so that time values aren't too big
        start_time = start_time - Constants.TIME_CENTER
        
        heart_rate_array = heart_rate_array.values.flatten()
        
        #data value frequency in Hertz
        data_frequency = heart_rate_array[0]

        heart_rate_array = heart_rate_array[1:]
        
        timestamps = [(start_time + i*(1/data_frequency)) for i in range(len(heart_rate_array))]
        heart_rate_array = np.stack([timestamps, heart_rate_array], axis=1)
        
        return heart_rate_array

    @staticmethod
    def write(heart_rate_collection, sleep_session_id):
        hr_output_path = HeartRateService.get_cropped_file_path(heart_rate_collection.subject_id, sleep_session_id)
        np.savetxt(hr_output_path, heart_rate_collection.data, fmt='%f')

    @staticmethod
    def crop(heart_rate_collection, interval):
        subject_id = heart_rate_collection.subject_id
        timestamps = heart_rate_collection.timestamps
        valid_indices = ((timestamps >= interval.start_time)
                         & (timestamps < interval.end_time)).nonzero()[0]

        cropped_data = heart_rate_collection.data[valid_indices, :]
        return HeartRateCollection(subject_id=subject_id, data=cropped_data)

    @staticmethod
    def get_cropped_file_path(subject_id, sleep_session_id):
        directory_path_string = str(subject_id) + "/" + "sleepsession_" + str(sleep_session_id)
        
        subject_folder_path = Constants().CROPPED_FILE_PATH.joinpath(str(subject_id))
        # creating a subject folder if it doesn't already exist
        if not os.path.exists(subject_folder_path):
            os.mkdir(subject_folder_path)
            
        sleep_session_path = Constants().CROPPED_FILE_PATH.joinpath(directory_path_string)
        # creating a sleep session folder if it doesn't already exist
        if not os.path.exists(sleep_session_path):
            os.mkdir(sleep_session_path)
        
        return Constants.CROPPED_FILE_PATH.joinpath(directory_path_string + "/cropped_hr.out")

    @staticmethod
    def get_raw_file_path(subject_id):
        subject_dir = utils.get_project_root().joinpath('USI Sleep/E4_Data/' + subject_id)
        session_dirs = os.listdir(subject_dir)
        session_dirs.sort()
        
        #Removing .DS_Store from the list of directories because we don't care about it
        session_dirs.remove('.DS_Store')
        
        #For now we are simply returning the first session
        #TODO: Return all directories, not only the first one
        return subject_dir.joinpath(session_dirs[2] + '/HR.csv')
