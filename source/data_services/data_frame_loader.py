from source.data_services.data_loader import DataLoader
from source.data_services.data_service import DataService
from source.preprocessing.built_service import BuiltService
from source.data_services.dataset import DataSet
from source.analysis.setup.feature_type import FeatureType
from multipledispatch import dispatch
from source.exception_logger import ExceptionLogger

import pandas as pd
import numpy as np
import sys


class DataFrameLoader(object):
    
    @staticmethod
    def load_epoched_dataframe(subject_id, session_id, feature_type, sleep_wake, dataset):
        feature = DataService.load_feature_raw(subject_id, session_id, feature_type, sleep_wake, dataset)
        feature_columns = DataLoader.load_columns(feature_type, sleep_wake, dataset)
        feature_df = pd.DataFrame(feature)
        feature_df.columns = feature_columns
        return feature_df
    
    @staticmethod 
    @dispatch(str, str, list, object, object)
    def load_feature_dataframe(subject_id, session_id, feature_types, sleep_wake, dataset):
        i = 0
        for feature_type in feature_types:
            feature_df = DataFrameLoader.load_epoched_dataframe(subject_id, session_id, feature_type, sleep_wake, dataset)
            
            if i == 0:
                final_features = feature_df
            else:
                final_features = pd.merge(final_features, feature_df, how="inner", on=["epoch_timestamp"])
            i += 1
        if 'subject_id' not in final_features:
            final_features.insert(1, 'subject_id', str(subject_id))
        if 'session_id' not in final_features:
            final_features.insert(2, 'session_id', str(session_id))
        return final_features
    
    @staticmethod
    @dispatch(str, list, object, object)
    def load_feature_dataframe(subject_id, feature_types, sleep_wake, dataset): 
        session_ids = BuiltService.get_built_sleepsession_ids(subject_id, feature_types[0], sleep_wake, dataset)

        feature_shape = DataFrameLoader.__get_dataframe_shape(subject_id, feature_types, sleep_wake, dataset)
        
        stacked_feature_df = pd.DataFrame(np.zeros(feature_shape))
        
        current_height = 0
        for session_id in session_ids:
            try:
                feature_df = DataFrameLoader.load_feature_dataframe(subject_id, session_id, feature_types, sleep_wake, dataset)
                columns = feature_df.columns
                feature_height = feature_df.shape[0]
                
                stacked_feature_df.iloc[current_height:(current_height + feature_height),:] = feature_df
                
                current_height += feature_height
            except:
                ExceptionLogger.append_exception(subject_id, session_id, "DataLoader", dataset.name, sys.exc_info()[0])
                print("Skip subject ", str(subject_id), ", session ", str(session_id), " due to ", sys.exc_info()[0])
        
        stacked_feature_df.columns = columns
        return stacked_feature_df
    
    @staticmethod
    @dispatch(list, object, object)
    def load_feature_dataframe(feature_types, sleep_wake, dataset):
        
        subject_ids = BuiltService.get_built_subject_ids(feature_types[0], sleep_wake, dataset)

        feature_shape = DataFrameLoader.__get_dataframe_shape(feature_types, sleep_wake, dataset)
        
        stacked_feature_df = pd.DataFrame(np.zeros(feature_shape))
        
        current_height = 0
        for subject_id in subject_ids:
            
            try:
                feature_df = DataFrameLoader.load_feature_dataframe(subject_id, feature_types, sleep_wake, dataset)
                columns = feature_df.columns
                feature_height = feature_df.shape[0]
                
                stacked_feature_df.iloc[current_height:(current_height + feature_height),:] = feature_df
                
                current_height += feature_height
            except:
                ExceptionLogger.append_exception(subject_id, 'N/A', "DataLoader", dataset.name, sys.exc_info()[0])
                print("Skip subject ", str(subject_id), " due to ", sys.exc_info()[0])

                
        stacked_feature_df.columns = columns
        return stacked_feature_df
    
    @staticmethod
    @dispatch(list, object, list)
    def load_feature_dataframe(feature_types, sleep_wake, datasets):

        feature_shape = DataFrameLoader.__get_dataframe_shape(feature_types, sleep_wake, datasets)
        
        stacked_feature_df = pd.DataFrame(np.zeros(feature_shape))
        
        current_height = 0
        for dataset in datasets:
            
            feature_df = DataFrameLoader.load_feature_dataframe(feature_types, sleep_wake, dataset)
            columns = feature_df.columns
            feature_height = feature_df.shape[0]
            
            stacked_feature_df.iloc[current_height:(current_height + feature_height),:] = feature_df
            
            current_height += feature_height
        
        stacked_feature_df.columns = columns
        return stacked_feature_df
            
    
    @staticmethod
    @dispatch(str, list, object, object)
    def __get_dataframe_shape(subject_id, feature_types, sleep_wake, dataset):
        session_ids = BuiltService.get_built_sleepsession_ids(subject_id, feature_types[0], sleep_wake, dataset)
        
        stacked_height = 0
        for i in range(len(session_ids)):
            try:
                feature_shape = DataFrameLoader.load_feature_dataframe(subject_id, session_ids[i], feature_types, sleep_wake, dataset).to_numpy().shape
                    
                feature_height = feature_shape[0]
                feature_width = feature_shape[1]            
    
                stacked_height = stacked_height + feature_height
            except:
                pass
        
        return (stacked_height, feature_width)
    
    @staticmethod
    @dispatch(list, object, object)
    def __get_dataframe_shape(feature_types, sleep_wake, dataset):

        subject_ids = BuiltService.get_built_subject_ids(feature_types[0], sleep_wake, dataset)

        stacked_height = 0
        for i in range(len(subject_ids)):
            try:
                feature_shape = DataFrameLoader.__get_dataframe_shape(subject_ids[i], feature_types, sleep_wake, dataset)
                feature_height = feature_shape[0]
                feature_width = feature_shape[1]

                stacked_height = stacked_height + feature_height
            except:
                pass
        
        return (stacked_height, feature_width)

    @staticmethod
    @dispatch(list, object, list)
    def __get_dataframe_shape(feature_types, sleep_wake, datasets):
        
        i = 0
        for dataset in datasets:
            
            feature_shape = DataFrameLoader.__get_dataframe_shape(feature_types, sleep_wake, dataset)
            feature_height = feature_shape[0]
            feature_width = feature_shape[1]
            
            if i == 0:
                stacked_height = feature_height
            else:
                # This has quite a bad runtime, might want to make it faster at some point
                stacked_height = stacked_height + feature_height
            i += 1
        
        return (stacked_height, feature_width)