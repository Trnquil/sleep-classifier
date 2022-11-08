import os


from source.constants import Constants
from source.analysis.setup.sleep_session_service import SleepSessionService

class BuiltService(object):
    
    # returns only subject and sleepsession ids for which there is data
    @staticmethod
    def get_built_subject_and_sleepsession_ids(path):
        subject_to_session_dictionary = {}
        
        for subject_id in Constants.SUBJECT_IDS:
                if subject_id in os.listdir(path):
                    session_dirs = os.listdir(path.joinpath(subject_id))
                    for session_id in SleepSessionService.get_starttime_ordered_ids(subject_id):
                        if session_id in session_dirs:
                            if subject_id not in subject_to_session_dictionary.keys():
                                subject_to_session_dictionary[subject_id] = []
                                
                            subject_to_session_dictionary[subject_id].append(session_id)
        return subject_to_session_dictionary  

    # returns only sleepsession ids for which there is data
    @staticmethod 
    def get_built_sleepsession_ids(subject_id, path):
        return BuiltService.get_built_subject_and_sleepsession_ids(path)[subject_id]    

    # returns only subject ids for which there is data
    @staticmethod 
    def get_built_subject_ids(path):
        return list(BuiltService.get_built_subject_and_sleepsession_ids(path).keys())
    
    # returns only subject ids for which there is data
    @staticmethod 
    def get_built_subject_and_sleepsession_count(path):
        count = 0
        for subject_id in BuiltService.get_built_subject_ids(path):
            for session_id in BuiltService.get_built_sleepsession_ids(subject_id, path):
                count += 1
        return count