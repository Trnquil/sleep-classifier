import sys
sys.path.insert(1, '../..') # tells system where project root is

from source.analysis.setup.feature_type import FeatureType
from source.preprocessing.built_service import BuiltService
from source.analysis.setup.sleep_session_service import SleepSessionService
from source.data_services.data_loader import DataLoader
from source.constants import Constants

from multipledispatch import dispatch
import numpy as np

class DataService(object):
    
    @staticmethod
    @dispatch(str, str, object)
    def load_feature_raw(subject_id, session_id, feature_type):
        
        if feature_type.name in FeatureType.get_cropped_names(): 
            feature = DataLoader.load_cropped(subject_id, session_id, feature_type).data
            
        elif feature_type.name in FeatureType.get_epoched_names(): 
            feature = DataLoader.load_epoched(subject_id, session_id, feature_type)
            
        elif feature_type.name in FeatureType.epoched.name: 
            feature = DataLoader.load_epoched(subject_id, session_id, feature_type).values
            
        elif feature_type.name in FeatureType.get_nightly_names(): 
            feature = DataLoader.load_nightly(subject_id, session_id, feature_type).to_numpy()
            
        elif FeatureType.sleep_quality.name == feature_type.name:
            feature = np.array([SleepSessionService.load_sleepquality(subject_id, session_id)]).reshape(1,1)
            
        else:
            raise Exception("FeatureType unknown to DataService")
        return feature
    
    @staticmethod
    @dispatch(str, object)
    def load_feature_raw(subject_id, feature_type):
        if feature_type.name in FeatureType.get_cropped_names():   
            session_ids = BuiltService.get_built_sleepsession_ids(subject_id, Constants.CROPPED_FILE_PATH)
        else:
            session_ids = BuiltService.get_built_sleepsession_ids(subject_id, Constants.EPOCHED_FILE_PATH)
            
        feature_shape = DataService.__get_feature_shape(subject_id, feature_type)
        
        stacked_feature = np.zeros(feature_shape)
        
        current_height = 0
        for session_id in session_ids:
            
            feature = DataService.load_feature_raw(subject_id, session_id, feature_type)
            feature_height = feature.shape[0]
            
            stacked_feature[current_height:(current_height + feature_height)][:] = feature
            
            current_height += feature_height
        
        return stacked_feature
    
    @staticmethod
    @dispatch(object)
    def load_feature_raw(feature_type):
        
        if feature_type.name in FeatureType.get_cropped_names():     
            subject_ids = BuiltService.get_built_subject_ids(Constants.CROPPED_FILE_PATH)
        else:
            subject_ids = BuiltService.get_built_subject_ids(Constants.EPOCHED_FILE_PATH)
            
        feature_shape = DataService.__get_feature_shape(feature_type)
        
        stacked_feature = np.zeros(feature_shape)
        
        current_height = 0
        for subject_id in subject_ids:
            
            feature = DataService.load_feature_raw(subject_id, feature_type)
            feature_height = feature.shape[0]
            
            stacked_feature[current_height:(current_height + feature_height)][:] = feature
            
            current_height += feature_height
        
        return stacked_feature
    
    @staticmethod
    @dispatch(str, object)
    def __get_feature_shape(subject_id, feature_type):
        if feature_type.name in FeatureType.get_cropped_names():     
            session_ids = BuiltService.get_built_sleepsession_ids(subject_id, Constants.CROPPED_FILE_PATH)
        else:
            session_ids = BuiltService.get_built_sleepsession_ids(subject_id, Constants.EPOCHED_FILE_PATH)
        
        for i in range(len(session_ids)):
            
            feature_shape = DataService.load_feature_raw(subject_id, session_ids[i], feature_type).shape
                
            feature_height = feature_shape[0]
            feature_width = feature_shape[1]
            
            if i == 0:
                stacked_height = feature_height
            else:
                # This has quite a bad runtime, might want to make it faster at some point
                stacked_height = stacked_height + feature_height
        
        return (stacked_height, feature_width)
    
    @staticmethod
    @dispatch(object)
    def __get_feature_shape(feature_type):
        if feature_type.name in FeatureType.get_cropped_names():   
            subject_ids = BuiltService.get_built_subject_ids(Constants.CROPPED_FILE_PATH)
        else:
            subject_ids = BuiltService.get_built_subject_ids(Constants.EPOCHED_FILE_PATH)
        
        for i in range(len(subject_ids)):
            
            feature_shape = DataService.__get_feature_shape(subject_ids[i], feature_type)
            feature_height = feature_shape[0]
            feature_width = feature_shape[1]
            
            if i == 0:
                stacked_height = feature_height
            else:
                # This has quite a bad runtime, might want to make it faster at some point
                stacked_height = stacked_height + feature_height
        
        return (stacked_height, feature_width)
                
    


                