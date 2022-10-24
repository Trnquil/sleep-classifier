import sys
sys.path.insert(1, '..') # tells system where project root is

from source.analysis.setup.feature_type import FeatureType
from source.preprocessing.path_service import PathService
from source.preprocessing.collection import Collection

from source.analysis.setup.sleep_session_service import SleepSessionService
from multipledispatch import dispatch
from source.analysis.setup.feature_type import FeatureType
from source.preprocessing.built_service import BuiltService

import numpy as np
import pandas as pd

class DataService(object):
    
    @staticmethod
    @dispatch(str, str, object)
    def load_feature_raw(subject_id, session_id, feature_type):
        
        if feature_type.name in FeatureType.get_cropped_names(): 
            feature = DataService.load_cropped(subject_id, session_id, feature_type).data
            
        elif feature_type.name in FeatureType.get_epoched_names(): 
            feature = DataService.load_epoched(subject_id, session_id, feature_type)
            
        elif FeatureType.sleep_quality.name == feature_type.name:
            feature = np.array([SleepSessionService.load_sleepquality(subject_id, session_id)]).reshape(1)
            
        else:
            raise Exception("FeatureType unknown to DataService")
        return feature
    
    @staticmethod
    @dispatch(str, object)
    def load_feature_raw(subject_id, feature_type):
        
        session_ids = BuiltService.get_built_sleepsession_ids(subject_id)
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
        
        subject_ids = BuiltService.get_built_subject_ids()
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
    def load_cropped(subject_id, session_id, feature_type):
        file_path = PathService.get_cropped_file_path(subject_id, session_id, feature_type)
        values = pd.read_csv(str(file_path), delimiter=" ").values
        return Collection(subject_id=subject_id, data=values)
    
    @staticmethod
    def write_cropped(collection, session_id, feature_type):
        output_path = PathService.get_cropped_file_path(collection.subject_id, session_id, feature_type)
        np.savetxt(output_path, collection.data, fmt='%f')
    
    @staticmethod
    def load_epoched(subject_id, session_id, feature_type):
        feature_path = PathService.get_epoched_file_path(subject_id, session_id, feature_type)
        feature = pd.read_csv(str(feature_path)).values
        return feature
    
    @staticmethod
    def write_epoched(subject_id, session_id, feature, feature_type):
        feature_path = PathService.get_epoched_file_path(subject_id, session_id, feature_type)
        np.savetxt(feature_path, feature, fmt='%f')
        
    @staticmethod
    def load_nightly():
        nightly_feature_path = PathService.get_nightly_file_path()
        nightly_feature_dataframe = pd.read_csv(str(nightly_feature_path))
        return nightly_feature_dataframe
    
    @staticmethod
    def write_nightly(nightly_dataframe):
        nightly_feature_path = PathService.get_nightly_file_path()
        nightly_dataframe.to_csv(nightly_feature_path, index=False)
    
    @staticmethod
    @dispatch(str, object)
    def __get_feature_shape(subject_id, feature_type):
        session_ids = BuiltService.get_built_sleepsession_ids(subject_id)
        
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
        subject_ids = BuiltService.get_built_subject_ids()
        
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
                
    


                