
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


class SleepSessionService(object):
    
    @staticmethod
    def get_file_path():
        return utils.get_project_root().joinpath('USI Sleep/Selfreports/final_labels.csv')
    
    @staticmethod
    def load(subject_id):
        selfreports_file = SleepSessionService.get_file_path()
        selfreports_array = pd.read_csv(str(selfreports_file))
        
        # We only keep the labels that we are interested in
        selfreports_array = selfreports_array[selfreports_array['User'].eq(subject_id)]
        selfreports_array = selfreports_array[['Start_Timestamp', 'End_Timestamp', 'Avg_SleepQuality_label']]
        
        sleepsessions = []
        for index, selfreports_row in selfreports_array.iterrows():
            start_timestamp = SleepSessionService.convert_to_centered_time(selfreports_row[0])
            end_timestamp = SleepSessionService.convert_to_centered_time(selfreports_row[1])
            sleepquality = int(selfreports_row[2])
            
            sleepsession = SleepSession(sleepquality, start_timestamp, end_timestamp)
            sleepsessions.append(sleepsession)
        
        return sleepsessions
    
    
    @staticmethod
    def convert_to_centered_time(timestamp):
        timestamp = datetime.strptime(timestamp, '%Y:%m:%d %H:%M:%S:%f')
        
        #unix epoch timestamp
        timestamp = time.mktime(timestamp.timetuple())
        
        #centering unix epoch timestamp so that it isn't too large
        return timestamp - Constants.TIME_CENTER
    
subject_id = ('S01')
sleepsessions = SleepSessionService.load(subject_id)

        
