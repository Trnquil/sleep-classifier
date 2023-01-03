from source.analysis.setup.feature_type import FeatureType
from source.preprocessing.built_service import BuiltService
from source.preprocessing.sleep_session_services.sleep_session_service import SleepSessionService
from source.data_services.data_loader import DataLoader
from source.constants import Constants
from source.data_services.dataset import DataSet

from multipledispatch import dispatch
import numpy as np
import pandas as pd

class DataService(object):
    @staticmethod
    @dispatch(str, str, object, object, object)
    def load_feature_raw(subject_id, session_id, feature_type, sleep_wake, dataset):
        
        if feature_type.name in FeatureType.get_cropped_names() and dataset.name == DataSet.usi.name: 
            feature = DataLoader.load_cropped(subject_id, session_id, feature_type, sleep_wake, dataset).data
            
        elif feature_type.name in FeatureType.get_epoched_names() or feature_type.name == FeatureType.epoched.name: 
            feature = DataLoader.load_epoched(subject_id, session_id, feature_type, sleep_wake, dataset).values
            
        elif feature_type.name in FeatureType.get_cluster_names(): 
            feature = DataLoader.load_cluster(subject_id, session_id, feature_type, sleep_wake, dataset).values
            
        elif feature_type.name in FeatureType.get_nightly_names(): 
            feature = DataLoader.load_nightly(subject_id, session_id, feature_type, sleep_wake, dataset).to_numpy()
            
        elif FeatureType.sleep_quality.name == feature_type.name:
            feature = np.array([SleepSessionService.load_sleepquality(subject_id, session_id, dataset)]).reshape(1,1)
            
        else:
            raise Exception("FeatureType unknown to DataService")
        return feature
    
    @staticmethod
    @dispatch(str, object, object, object)
    def load_feature_raw(subject_id, feature_type, sleep_wake, dataset):
        session_ids = BuiltService.get_built_sleepsession_ids(subject_id, feature_type, sleep_wake, dataset)

            
        feature_shape = DataService.__get_feature_shape(subject_id, feature_type, sleep_wake, dataset)
        
        stacked_feature = np.zeros(feature_shape)
        
        current_height = 0
        for session_id in session_ids:
            
            feature = DataService.load_feature_raw(subject_id, session_id, feature_type, sleep_wake, dataset)
            feature_height = feature.shape[0]
            
            stacked_feature[current_height:(current_height + feature_height)][:] = feature
            
            current_height += feature_height
        
        return stacked_feature
    
    @staticmethod
    @dispatch(object, object, object)
    def load_feature_raw(feature_type, sleep_wake, dataset):
        
        subject_ids = BuiltService.get_built_subject_ids(feature_type, sleep_wake, dataset)

        feature_shape = DataService.__get_feature_shape(feature_type, sleep_wake, dataset)
        
        stacked_feature = np.zeros(feature_shape)
        
        current_height = 0
        for subject_id in subject_ids:
            
            feature = DataService.load_feature_raw(subject_id, feature_type, sleep_wake, dataset)
            feature_height = feature.shape[0]
            
            stacked_feature[current_height:(current_height + feature_height)][:] = feature
            
            current_height += feature_height
        
        return stacked_feature
    
    @staticmethod
    @dispatch(str, object, object, object)
    def __get_feature_shape(subject_id, feature_type, sleep_wake, dataset):
  
        session_ids = BuiltService.get_built_sleepsession_ids(subject_id, feature_type, sleep_wake, dataset)

        for i in range(len(session_ids)):
            
            feature_shape = DataService.load_feature_raw(subject_id, session_ids[i], feature_type, sleep_wake, dataset).shape
                
            feature_height = feature_shape[0]
            feature_width = feature_shape[1]
            
            if i == 0:
                stacked_height = feature_height
            else:
                # This has quite a bad runtime, might want to make it faster at some point
                stacked_height = stacked_height + feature_height
        
        return (stacked_height, feature_width)
    
    @staticmethod
    @dispatch(object, object, object)
    def __get_feature_shape(feature_type, sleep_wake, dataset):

        subject_ids = BuiltService.get_built_subject_ids(feature_type, sleep_wake, dataset)

        
        for i in range(len(subject_ids)):
            
            feature_shape = DataService.__get_feature_shape(subject_ids[i], feature_type, sleep_wake, dataset)
            feature_height = feature_shape[0]
            feature_width = feature_shape[1]
            
            if i == 0:
                stacked_height = feature_height
            else:
                # This has quite a bad runtime, might want to make it faster at some point
                stacked_height = stacked_height + feature_height
        
        return (stacked_height, feature_width)
                
    


                