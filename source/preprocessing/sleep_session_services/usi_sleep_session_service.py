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
    def get_selfreported_file_path():
        return utils.get_project_root().joinpath('data/USI Sleep/Selfreports/final_labels.csv')
    
    def get_file_path():
        return utils.get_project_root().joinpath('data/USI Sleep/morningsurvey_usi.csv')
    
    @staticmethod
    def load_sleepquality(subject_id, session_id):
        selfreports_file = UsiSleepSessionService.get_selfreported_file_path()
        selfreports_array = pd.read_csv(str(selfreports_file))
        
        selfreports_array = selfreports_array[selfreports_array['User'].eq(subject_id)]
        selfreports_array = selfreports_array[selfreports_array['End_Date'].eq(session_id.replace("-", ":"))]
        
        selfreports_array = selfreports_array[['Avg_SleepQuality_label']]
        
        sleepquality = int(selfreports_array.iloc[0])

        return sleepquality
    
        
    @staticmethod
    def load_selfreported_sleep(subject_id):
        selfreports_file = UsiSleepSessionService.get_selfreported_file_path()
        selfreports_array = pd.read_csv(str(selfreports_file))
        
        # making sure we only work with data from the requested user (specified by subject_id)
        selfreports_array = selfreports_array[selfreports_array['User'].eq(subject_id)]
        
        # We only keep the labels that we are interested in
        selfreports_array = selfreports_array[['SessionID', 'Start_Timestamp', 'End_Timestamp', 'Avg_SleepQuality_label']].dropna()
        
        sleepsessions = []
        for index, selfreports_row in selfreports_array.iterrows():
            
            sleep_time = UsiSleepSessionService.convert_to_unix_timestamp(selfreports_row[1])
            awake_time = UsiSleepSessionService.convert_to_unix_timestamp(selfreports_row[2])
            
            session_id = datetime.utcfromtimestamp(awake_time).strftime('%Y-%m-%d')
            start_timestamp = sleep_time - Constants.TIME_CENTER_USI
            end_timestamp = awake_time - Constants.TIME_CENTER_USI
            sleepquality = int(selfreports_row[3])
            
            sleepsession = SleepSession(session_id, sleepquality, start_timestamp, end_timestamp)
            sleepsessions.append(sleepsession)
        
        sleepsessions.sort(key=lambda sleepsession: sleepsession.start_timestamp)
        return sleepsessions
    
    @staticmethod
    def load_sleep(subject_id):
        sleep_wake_file = UsiSleepSessionService.get_file_path()
        sleep_wake_array = pd.read_csv(str(sleep_wake_file))
        
        # making sure we only work with data from the requested user (specified by subject_id)
        sleep_wake_array = sleep_wake_array[sleep_wake_array['userID'].eq(subject_id)]
        
        # We only keep the labels that we are interested in
        sleep_wake_array = sleep_wake_array[['awake', 'sleep', 'awake_next', 'sleepRecovery']].dropna()
        
        sleepsessions = []
        sleep_wake_rows = list(sleep_wake_array.iterrows())
        
        # Getting rid of the indexes at x[0]
        sleep_wake_rows = [x[1] for x in sleep_wake_rows]
        
        for i in range(1, len(sleep_wake_rows)):
            
            if(sleep_wake_rows[i - 1][2] != sleep_wake_rows[i][0]):
                continue
            
            awake_time = int(sleep_wake_rows[i][0])
            
            # We need sleep time from last night
            sleep_time = int(sleep_wake_rows[i - 1][1])
            
            session_id = datetime.utcfromtimestamp(awake_time).strftime('%Y-%m-%d')
            start_timestamp = sleep_time - Constants.TIME_CENTER_USI
            end_timestamp = awake_time - Constants.TIME_CENTER_USI
            

            sleepquality = int(sleep_wake_rows[i][3])
            
            sleepsession = SleepSession(session_id, sleepquality, start_timestamp, end_timestamp)
            sleepsessions.append(sleepsession)
        
        sleepsessions.sort(key=lambda sleepsession: sleepsession.start_timestamp)
        return sleepsessions
    
    @staticmethod
    def load_wake(subject_id):
        sleep_wake_file = UsiSleepSessionService.get_file_path()
        sleep_wake_array = pd.read_csv(str(sleep_wake_file))
        
        # making sure we only work with data from the requested user (specified by subject_id)
        sleep_wake_array = sleep_wake_array[sleep_wake_array['userID'].eq(subject_id)]
        
        # We only keep the labels that we are interested in
        sleep_wake_array = sleep_wake_array[['awake', 'sleep', 'sleepRecovery']].dropna()
        
        sleepsessions = []
        sleep_wake_rows = list(sleep_wake_array.iterrows())
        
        # Getting rid of the indexes at x[0]
        sleep_wake_rows = [x[1] for x in sleep_wake_rows]
        
        for i in range(1, len(sleep_wake_rows)):
            
            # We take the awake time from last morning
            awake_time = int(sleep_wake_rows[i - 1][0])
            
            # We take the sleep time from last night
            sleep_time = int(sleep_wake_rows[i - 1][1])
            
            # We take the awake time from last morning
            actual_awake_time = int(sleep_wake_rows[i][0])
            
            session_id = datetime.utcfromtimestamp(actual_awake_time).strftime('%Y-%m-%d')
            start_timestamp = awake_time - Constants.TIME_CENTER_USI
            end_timestamp = sleep_time - Constants.TIME_CENTER_USI
            sleepquality = int(sleep_wake_rows[i][2])
            
            sleepsession = SleepSession(session_id, sleepquality, start_timestamp, end_timestamp)
            sleepsessions.append(sleepsession)
        
        sleepsessions.sort(key=lambda sleepsession: sleepsession.start_timestamp)
        return sleepsessions
        
    @staticmethod
    def convert_to_unix_timestamp(timestamp):
        timestamp = datetime.strptime(timestamp, '%Y:%m:%d %H:%M:%S:%f')
        
        #unix epoch timestamp
        timestamp = time.mktime(timestamp.timetuple())
        
        #centering unix epoch timestamp so that it isn't too large
        return timestamp