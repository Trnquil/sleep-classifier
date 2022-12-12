from source import utils
from source.constants import Constants
from source.exception_logger import ExceptionLogger
from source.data_services.dataset import DataSet
from source.analysis.setup.sleep_session import SleepSession

import json
import sys
from datetime import datetime  
import pandas as pd

class MssSleepSessionService(object):
    
    @staticmethod
    def get_file_path():
        return utils.get_project_root().joinpath('data/MS Sleep/wake_sleep.json')
    
    @staticmethod
    def get_morning_survey_path(subject_id):
        return utils.get_project_root().joinpath("data/MS Sleep/Sensor and smartphone data export/data_max/" + 
                                                 str(subject_id) + "/morningsurvey.csv")
    
    @staticmethod
    def load_sleepquality(subject_id, session_id):
        wake_sleep_file = open(MssSleepSessionService.get_file_path())
        wake_sleep_dict = json.load(wake_sleep_file)
        wake_sleep_dict = wake_sleep_dict[subject_id]
        
        # We check which timestamp is lower and set it to start_timestamp and the other to end_timestamp
        if(wake_sleep_dict[session_id][0] < wake_sleep_dict[session_id][1]):
            start_timestamp = wake_sleep_dict[session_id][0]
            end_timestamp = wake_sleep_dict[session_id][1]
        else:
            start_timestamp = wake_sleep_dict[session_id][1]
            end_timestamp = wake_sleep_dict[session_id][0]
            
        sleepquality = MssSleepSessionService.get_sleepquality(subject_id, end_timestamp)
        return int(sleepquality)
        
        
    @staticmethod
    def load(subject_id):
        wake_sleep_file = open(MssSleepSessionService.get_file_path())
        wake_sleep_dict = json.load(wake_sleep_file)
        wake_sleep_dict = wake_sleep_dict[subject_id]
        
        sleepsessions = []
        
        for session_id in wake_sleep_dict.keys():
            try:
                # If the timestamps don't exist, we continue with the next session
                if wake_sleep_dict[session_id][0] is None or wake_sleep_dict[session_id][1] is None:
                    continue
                # We check which timestamp is lower and set it to start_timestamp and the other to end_timestamp
                if(wake_sleep_dict[session_id][0] < wake_sleep_dict[session_id][1]):
                    start_timestamp = wake_sleep_dict[session_id][0]
                    end_timestamp = wake_sleep_dict[session_id][1]
                else:
                    start_timestamp = wake_sleep_dict[session_id][1]
                    end_timestamp = wake_sleep_dict[session_id][0]
                
                
                end_timestamp_centered = end_timestamp - Constants.TIME_CENTER_MSS
                start_timestamp_centered = start_timestamp - Constants.TIME_CENTER_MSS
                sleepquality = MssSleepSessionService.get_sleepquality(subject_id, end_timestamp)
                
                sleep_session = SleepSession(session_id, sleepquality, start_timestamp_centered, end_timestamp_centered)
                sleepsessions.append(sleep_session)
                
            except:
                ExceptionLogger.append_exception(subject_id, session_id, "MSS SleepSession", DataSet.mss.name, sys.exc_info()[0])
                print("Skip subject ", str(subject_id), " due to ", sys.exc_info()[0])
                
        return sleepsessions
                
                
    @staticmethod
    def get_sleepquality(subject_id, end_timestamp):
        sleep_end_date = datetime.fromtimestamp(end_timestamp).strftime("%Y-%m-%d")  
        morning_survey_df = pd.read_csv(MssSleepSessionService.get_morning_survey_path(subject_id))
        for index, night_row in morning_survey_df.iterrows():
            morning_survey_date = datetime.fromtimestamp(night_row['timestamp']).strftime("%Y-%m-%d")  
            if(morning_survey_date == sleep_end_date):
                sleepquality = night_row['feeling']
                return sleepquality
