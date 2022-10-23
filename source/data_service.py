import sys
sys.path.insert(1, '..') # tells system where project root is

from source.analysis.setup.feature_type import FeatureType
from source.preprocessing.motion.motion_service import MotionService
from source.preprocessing.heart_rate.heart_rate_service import HeartRateService
from source.preprocessing.activity_count.activity_count_service import ActivityCountService
from source.preprocessing.clustering.clustering_service import ClusteringService
from source.preprocessing.time.time_based_feature_service import TimeBasedFeatureService
from source.preprocessing.heart_rate.heart_rate_feature_service import HeartRateFeatureService
from source.preprocessing.activity_count.activity_count_feature_service import ActivityCountFeatureService
from source.preprocessing.nightly_feature_service import NightlyFeatureService
from source.analysis.setup.sleep_session_service import SleepSessionService

from source.analysis.setup.subject_builder import SubjectBuilder
from multipledispatch import dispatch

import numpy as np
import pandas as pd

class DataService(object):
    
    @staticmethod
    @dispatch(str, str, object)
    def load_feature_raw(subject_id, session_id, feature_type):
        
        if FeatureType.cropped_motion.name == feature_type.name: 
            feature = MotionService.load_cropped(subject_id, session_id).data
        elif FeatureType.cropped_heart_rate.name == feature_type.name:
            feature = HeartRateService.load_cropped(subject_id, session_id).data
        elif FeatureType.cropped_count.name == feature_type.name:
            feature  = ActivityCountService.load_cropped(subject_id, session_id).data
        elif FeatureType.epoched_cluster.name == feature_type.name: 
            feature = ClusteringService.load(subject_id, session_id)
        elif FeatureType.epoched_cosine.name == feature_type.name:
            feature = TimeBasedFeatureService.load_cosine(subject_id, session_id)
        elif FeatureType.epoched_time.name == feature_type.name:
            feature  = TimeBasedFeatureService.load_time(subject_id, session_id)
        elif FeatureType.epoched_heart_rate.name == feature_type.name:
            feature = HeartRateFeatureService.load(subject_id, session_id)
        elif FeatureType.epoched_count.name == feature_type.name:
            feature = ActivityCountFeatureService.load(subject_id, session_id)
        elif FeatureType.nightly.name == feature_type.name:
            feature = NightlyFeatureService.load(subject_id, session_id).to_numpy()
        elif FeatureType.sleep_quality.name == feature_type.name:
            feature = np.array([SleepSessionService.load_sleepquality(subject_id, session_id)]).reshape(1)
        else:
            raise Exception("FeatureType unknown to DataService")
        return feature
    
    @staticmethod
    @dispatch(str, object)
    def load_feature_raw(subject_id, feature_type):
        
        session_ids = SubjectBuilder.get_built_sleepsession_ids(subject_id)
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
        
        subject_ids = SubjectBuilder.get_built_subject_ids()
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
        session_ids = SubjectBuilder.get_built_sleepsession_ids(subject_id)
        
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
        subject_ids = SubjectBuilder.get_built_subject_ids()
        
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
                
    


                