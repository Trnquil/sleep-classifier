import os


from source.constants import Constants
from source.analysis.setup.sleep_session_service import SleepSessionService

class BuiltService(object):
    
    # returns only subject and sleepsession ids for which there is data
    @staticmethod
    def get_built_subject_and_sleepsession_ids():
        subject_to_session_dictionary = {}
        
        subject_dirs = os.listdir(Constants.CROPPED_FILE_PATH)
        subject_dirs.sort()
        for subject_id in os.listdir(Constants.CROPPED_FILE_PATH):
                if subject_id in Constants.SUBJECT_IDS:
                    session_dirs = os.listdir(Constants.CROPPED_FILE_PATH.joinpath(subject_id))
                    for session_id in SleepSessionService.get_starttime_ordered_ids(subject_id):
                        if session_id in session_dirs:
                            if subject_id not in subject_to_session_dictionary.keys():
                                subject_to_session_dictionary[subject_id] = []
                                
                            subject_to_session_dictionary[subject_id].append(session_id)
        return subject_to_session_dictionary  

    # returns only sleepsession ids for which there is data
    @staticmethod 
    def get_built_sleepsession_ids(subject_id):
        return BuiltService.get_built_subject_and_sleepsession_ids()[subject_id]    

    # returns only subject ids for which there is data
    @staticmethod 
    def get_built_subject_ids():
        return list(BuiltService.get_built_subject_and_sleepsession_ids().keys())
    
    # returns only subject ids for which there is data
    @staticmethod 
    def get_built_subject_and_sleepsession_count():
        count = 0
        for subject_id in BuiltService.get_built_subject_ids():
            for session_id in BuiltService.get_built_sleepsession_ids(subject_id):
                count += 1
        return count