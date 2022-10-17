import sys
# caution: path[0] is reserved for script path (or '' in REPL)
# We are inserting the sleepclassifier as a module 
sys.path.insert(1, '../../..')

import numpy as np
import pandas as pd
import os

from source import utils
from source.constants import Constants
from source.preprocessing.motion.motion_collection import MotionCollection
from source.preprocessing.path_service import PathService


class MotionService(object):

    @staticmethod
    def load_raw(subject_id):
        raw_motion_path = MotionService.get_raw_file_path(subject_id)
        motion_array = MotionService.load(raw_motion_path, delimiter=',')
        motion_array = utils.remove_repeats(motion_array)
        return MotionCollection(subject_id=subject_id, data=motion_array)

    @staticmethod
    def load_cropped(subject_id, session_id):
        cropped_motion_path = MotionService.get_cropped_file_path(subject_id, session_id)
        motion_array = pd.read_csv(str(cropped_motion_path), delimiter=' ', header=None)
        motion_array = motion_array.values
        return MotionCollection(subject_id=subject_id, data=motion_array)

    #delimiter doesn't do anything but I don't want to break the rest of the program
    @staticmethod
    def load(motion_file, delimiter=' '):
        motion_array = pd.read_csv(str(motion_file), delimiter=delimiter)
        
        #unix start time of the data
        start_time = motion_array.columns.values.tolist()
        start_time = float(start_time[0])
        
        #Let us center the time so that time values aren't too big
        start_time = start_time - Constants.TIME_CENTER
    
        motion_array = motion_array.values
        
        #data value frequency in Hertz
        data_frequency = motion_array[0][0]
        
        #We divide by 64 because acceleration is measured in 1/64g (more information in info.txt in the dataset)
        motion_array = motion_array[1:][:]/64
        
        #timestamps created at the intervals specified by data_frequency
        timestamps = [(start_time + i*(1/data_frequency)) for i in range(len(motion_array))]
        timestamps = np.array(timestamps).reshape((len(timestamps), 1))
        
        #There might be a faster way to implement this, but for now this should be ok
        motion_array = np.concatenate([timestamps, motion_array], axis=1)
        return motion_array                     
                                 
    @staticmethod
    def write(motion_collection, sleep_session_id):
        motion_output_path = MotionService.get_cropped_file_path(motion_collection.subject_id, sleep_session_id)
        np.savetxt(motion_output_path, motion_collection.data, fmt='%f')

    @staticmethod
    def crop(motion_collection, interval):
        subject_id = motion_collection.subject_id
        timestamps = motion_collection.timestamps
        valid_indices = ((timestamps >= interval.start_time)
                         & (timestamps < interval.end_time)).nonzero()[0]

        cropped_data = motion_collection.data[valid_indices, :]
        return MotionCollection(subject_id=subject_id, data=cropped_data)

    @staticmethod
    def get_cropped_file_path(subject_id, session_id):   
        return PathService.get_cropped_folder_path(subject_id, session_id) + "/cropped_motion.out"

    @staticmethod
    def get_raw_file_path(subject_id):
        return PathService.get_raw_folder_path(subject_id) + '/ACC.csv'

