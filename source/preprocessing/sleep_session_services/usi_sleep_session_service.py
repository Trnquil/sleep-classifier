from source import utils
import numpy as np
import pandas as pd
from datetime import datetime
import time

from source.constants import Constants
from source.analysis.setup.sleep_session import SleepSession
from source.preprocessing.collection import Collection

class UsiSleepSessionService(object):
    
    @staticmethod
    def get_file_path():
        return utils.get_project_root().joinpath('data/USI Sleep/Selfreports/final_labels.csv')
    
    @staticmethod
    def load_sleepquality(subject_id, session_id):
        selfreports_file = UsiSleepSessionService.get_file_path()
        selfreports_array = pd.read_csv(str(selfreports_file))
        
        selfreports_array = selfreports_array[selfreports_array['User'].eq(subject_id)]
        session_id_int = int(session_id[-2:])
        selfreports_array = selfreports_array[selfreports_array['SessionID'].eq(session_id_int)]
        
        selfreports_array = selfreports_array[['Avg_SleepQuality_label']]
        
        sleepquality = int(selfreports_array.iloc[0])

        return sleepquality
    
        
    @staticmethod
    def load(subject_id):
        selfreports_file = UsiSleepSessionService.get_file_path()
        selfreports_array = pd.read_csv(str(selfreports_file))
        
        # making sure we only work with data from the requested user (specified by subject_id)
        selfreports_array = selfreports_array[selfreports_array['User'].eq(subject_id)]
        
        # We only keep the labels that we are interested in
        selfreports_array = selfreports_array[['SessionID', 'Start_Timestamp', 'End_Timestamp', 'Avg_SleepQuality_label']].dropna()
        
        sleepsessions = []
        for index, selfreports_row in selfreports_array.iterrows():
            
            session_id = "SS_" + str(selfreports_row[0]).zfill(2)
            start_timestamp = UsiSleepSessionService.convert_to_centered_time(selfreports_row[1])
            end_timestamp = UsiSleepSessionService.convert_to_centered_time(selfreports_row[2])
            sleepquality = int(selfreports_row[3])
            
            sleepsession = SleepSession(session_id, sleepquality, start_timestamp, end_timestamp)
            sleepsessions.append(sleepsession)
        
        sleepsessions.sort(key=lambda sleepsession: sleepsession.start_timestamp)
        return sleepsessions
    
    @staticmethod
    def convert_to_centered_time(timestamp):
        timestamp = datetime.strptime(timestamp, '%Y:%m:%d %H:%M:%S:%f')
        
        #unix epoch timestamp
        timestamp = time.mktime(timestamp.timetuple())
        
        #centering unix epoch timestamp so that it isn't too large
        return timestamp - Constants.TIME_CENTER_USI