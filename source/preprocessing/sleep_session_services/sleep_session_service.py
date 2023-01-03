

import numpy as np

from source.preprocessing.collection import Collection
from source.preprocessing.sleep_session_services.usi_sleep_session_service import UsiSleepSessionService
from source.preprocessing.sleep_session_services.mss_sleep_session_service import MssSleepSessionService
from source.data_services.dataset import DataSet
from source.preprocessing.sleep_wake import SleepWake


class SleepSessionService(object):
    
    @staticmethod
    def load_sleepsession(subject_id, session_id, sleep_wake, dataset):
        if(dataset.name == DataSet.usi.name):
            if(sleep_wake.name == SleepWake.selfreported_sleep.name):
                sleepsessions = UsiSleepSessionService.load_selfreported_sleep(subject_id)
            elif(sleep_wake.name == SleepWake.sleep.name):
                sleepsessions = UsiSleepSessionService.load_sleep(subject_id)
            elif(sleep_wake.name == SleepWake.wake.name):
                sleepsessions = UsiSleepSessionService.load_wake(subject_id)
        elif(dataset.name == DataSet.mss.name):
            sleepsessions = MssSleepSessionService.load(subject_id)

        sleepsession = [session for session in sleepsessions if session.session_id == session_id][0]
        return sleepsession
      
    @staticmethod
    def load_sleepquality(subject_id, session_id, dataset):
        if(dataset.name == DataSet.usi.name):
            sleepquality = UsiSleepSessionService.load_sleepquality(subject_id, session_id)
        elif(dataset.name == DataSet.mss.name):
            sleepquality = MssSleepSessionService.load_sleepquality(subject_id, session_id)
        return sleepquality
    
    @staticmethod
    def get_starttime_ordered_ids(subject_id, sleep_wake, dataset):
        if(dataset.name == DataSet.usi.name):
            if(sleep_wake.name == SleepWake.selfreported_sleep.name):
                session_ids = [sleepsession.session_id for sleepsession in UsiSleepSessionService.load_selfreported_sleep(subject_id)]
            elif(sleep_wake.name == SleepWake.sleep.name):
                session_ids = [sleepsession.session_id for sleepsession in UsiSleepSessionService.load_sleep(subject_id)]
            elif(sleep_wake.name == SleepWake.wake.name):
                session_ids = [sleepsession.session_id for sleepsession in UsiSleepSessionService.load_wake(subject_id)]
        return session_ids
        

    # assigning features to each sleepsession based on the features' timestamps
    @staticmethod
    def assign_collection_to_sleepsession(collection, sleepsessions):
            
        sleepsession_tuples = []
        
        for sleepsession in sleepsessions:
            
            timestamps = collection.timestamps
            values = collection.values
            
            feature_dimension = values.shape[1]
            timestamped_feature = np.zeros((0, feature_dimension + 1), dtype='float')
        
            timestamp_index_start = 0
            
            # Going through timestamps until we've got one that is inside the current sleepsession
            while(timestamp_index_start + 1 < len(timestamps)
                  and timestamps[timestamp_index_start] < sleepsession.start_timestamp):
                timestamp_index_start += 1
                
            timestamp_index_end = timestamp_index_start
                
            # Going through the timestamps that are inside the sleepsession and appending features to the sleepsession
            while(timestamp_index_end + 1 < len(timestamps)
                  and timestamps[timestamp_index_end + 1] <= sleepsession.end_timestamp):
                timestamp_index_end += 1
            
            timestamped_feature = collection.data[timestamp_index_start:timestamp_index_end][:]
            sleepsession_tuple = (sleepsession, Collection(collection.subject_id, timestamped_feature, collection.data_frequency))
            sleepsession_tuples.append(sleepsession_tuple)
            
        
        return sleepsession_tuples
    
    @staticmethod
    def assert_inorder(sleepsessions):
        for i in range (len(sleepsessions) - 1):
            if not (sleepsessions[i].end_timestamp <= sleepsessions[i + 1].start_timestamp):
                return False
        return True