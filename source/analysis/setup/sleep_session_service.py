
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
# We are inserting the sleepclassifier as a module 
sys.path.insert(1, '../../..')

from source import utils
import numpy as np
import pandas as pd
from datetime import datetime
import time

from source.constants import Constants
from source.analysis.setup.sleep_session import SleepSession
from source.preprocessing.motion.motion_collection import MotionCollection


class SleepSessionService(object):
    
    @staticmethod
    def get_file_path():
        return utils.get_project_root().joinpath('data/USI Sleep/Selfreports/final_labels.csv')
    
    @staticmethod
    def load(subject_id):
        selfreports_file = SleepSessionService.get_file_path()
        selfreports_array = pd.read_csv(str(selfreports_file))
        
        # making sure we only work with data from the requested user (specified by subject_id)
        selfreports_array = selfreports_array[selfreports_array['User'].eq(subject_id)]
        
        # We only keep the labels that we are interested in
        selfreports_array = selfreports_array[['SessionID', 'Start_Timestamp', 'End_Timestamp', 'Avg_SleepQuality_label']].dropna()
        
        sleepsessions = []
        for index, selfreports_row in selfreports_array.iterrows():
            
            session_id = "SLEEP_SESSION_" + str(selfreports_row[0]).zfill(2)
            start_timestamp = SleepSessionService.convert_to_centered_time(selfreports_row[1])
            end_timestamp = SleepSessionService.convert_to_centered_time(selfreports_row[2])
            sleepquality = int(selfreports_row[3])
            
            sleepsession = SleepSession(session_id, sleepquality, start_timestamp, end_timestamp)
            sleepsessions.append(sleepsession)
        
        return sleepsessions
    
    @staticmethod
    def get_all_session_ids(subject_id):
        selfreports_file = SleepSessionService.get_file_path()
        selfreports_array = pd.read_csv(str(selfreports_file))
        
        # making sure we only work with data from the requested user (specified by subject_id)
        selfreports_array = selfreports_array[selfreports_array['User'].eq(subject_id)]
        
        # We only keep the labels that we are interested in
        selfreports_array = selfreports_array[['SessionID']].dropna()
        
        sleepsession_ids = []
        for index, selfreports_row in selfreports_array.iterrows():
            
            session_id = "SLEEP_SESSION_" + str(selfreports_row[0]).zfill(2)
            
            sleepsession_ids.append(session_id)
        
        return sleepsession_ids
    
    # assigning features to each sleepsession based on the features' timestamps
    @staticmethod
    def assign_collection_to_sleepsession(subject_id, collection):
        sleepsessions = SleepSessionService.load(subject_id)
        sleepsession_tuples = []
        
        for sleepsession in sleepsessions:
            
            timestamps = collection.timestamps
            values = collection.values
            
            feature_dimension = values.shape[1]
            timestamped_feature = np.zeros((0, feature_dimension + 1), dtype='float')
        
            timestamp_index_start = 0
            
            # Going through timestamps until we've got one that is inside the current sleepsession
            while(timestamp_index_start < len(timestamps)
                  and timestamps[timestamp_index_start] < sleepsession.start_timestamp):
                timestamp_index_start += 1
                
            timestamp_index_end = timestamp_index_start
                
            # Going through the timestamps that are inside the sleepsession and appending features to the sleepsession
            while(timestamp_index_end < len(timestamps)
                  and timestamps[timestamp_index_end + 1] <= sleepsession.end_timestamp):
                timestamp_index_end += 1
            
            timestamped_feature = collection.data[timestamp_index_start:timestamp_index_end][:]
            sleepsession_tuple = (sleepsession, MotionCollection(subject_id, timestamped_feature))
            sleepsession_tuples.append(sleepsession_tuple)
            
        
        return sleepsession_tuples
    
    @staticmethod
    def convert_to_centered_time(timestamp):
        timestamp = datetime.strptime(timestamp, '%Y:%m:%d %H:%M:%S:%f')
        
        #unix epoch timestamp
        timestamp = time.mktime(timestamp.timetuple())
        
        #centering unix epoch timestamp so that it isn't too large
        return timestamp - Constants.TIME_CENTER
    
    @staticmethod
    def assert_inorder(sleepsessions):
        for i in range (len(sleepsessions) - 1):
            if not (sleepsessions[i].end_timestamp <= sleepsessions[i + 1].start_timestamp):
                return False
        return True