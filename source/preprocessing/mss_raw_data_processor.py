from source import utils
from source.data_services.mss_loader import MssLoader
from source.preprocessing.sleep_session_services.sleep_session_service import SleepSessionService
from source.preprocessing.path_service import PathService
from source.preprocessing.activity_count.activity_count_service import ActivityCountService
from source.data_services.data_writer import DataWriter
from source.analysis.setup.feature_type import FeatureType
from source.exception_logger import ExceptionLogger
from source.data_services.dataset import DataSet

import sys
import numpy as np
import pandas as pd

class MssRawDataProcessor(object):
    
    @staticmethod
    def crop_all(subject_id):
        try:
        
            '''Loading Data'''
            motion_collection = MssLoader.load_raw_motion(subject_id)
            hr_collection = MssLoader.load_raw_hr(subject_id)
            
            
            '''splitting each collection into sleepsessions'''
            motion_sleepsession_tuples = SleepSessionService.assign_collection_to_sleepsession(subject_id, DataSet.mss, motion_collection)
            hr_sleepsession_tuples = SleepSessionService.assign_collection_to_sleepsession(subject_id, DataSet.mss, hr_collection)        
                    
    
            '''writing all the data to disk'''
            for i in range(len(motion_sleepsession_tuples)):
                try:
                    session_id = motion_sleepsession_tuples[i][0].session_id
                    
                    motion_collection = motion_sleepsession_tuples[i][1]
                    hr_collection = hr_sleepsession_tuples[i][1]
                                    
                    if(np.any(motion_collection.data)):
                        count_collection = ActivityCountService.build_activity_counts_without_matlab(subject_id, motion_collection.data)
                        DataWriter.write_cropped(count_collection, session_id, FeatureType.cropped_count, DataSet.mss)
                        DataWriter.write_cropped(motion_collection, session_id, FeatureType.cropped_motion, DataSet.mss)
                        
                    if(np.any(hr_collection.data)):
                        DataWriter.write_cropped(hr_collection, session_id, FeatureType.cropped_hr, DataSet.mss)
                        
                        
                except:
                    ExceptionLogger.append_exception(subject_id, session_id, "Cropped", DataSet.mss.name, sys.exc_info()[0])
                    print("Skip subject ", str(subject_id), ", session ", str(session_id), " due to ", sys.exc_info()[0])
        except:
            ExceptionLogger.append_exception(subject_id, "N/A", "Cropped", DataSet.mss.name, sys.exc_info()[0])
            print("Skip subject ", str(subject_id), " due to ", sys.exc_info()[0])
            
    @staticmethod
    def get_user_ids():
        path = utils.get_project_root().joinpath('data/MS Sleep/labels.csv')
        labels_df = pd.read_csv(path)
        user_ids_df = labels_df['userID']
        user_ids = user_ids_df.to_list()
        return user_ids
        
        
