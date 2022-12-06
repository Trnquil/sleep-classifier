import sys
sys.path.insert(1, '../..') # tells system where project root is

from source.data_services.data_loader import DataLoader
from source.data_services.data_service import DataService
from source.preprocessing.built_service import BuiltService
from source.data_services.dataset import DataSet
from source.analysis.setup.feature_type import FeatureType
from multipledispatch import dispatch
from source.exception_logger import ExceptionLogger

import pandas as pd
import numpy as np



class DataFrameLoader(object):
    
    @staticmethod
    def load_epoched_dataframe(subject_id, session_id, feature_type, dataset):
        feature = DataService.load_feature_raw(subject_id, session_id, feature_type, dataset)
        feature_columns = DataLoader.load_columns(feature_type, dataset)
        feature_df = pd.DataFrame(feature)
        feature_df.columns = feature_columns
        return feature_df
    
    @staticmethod 
    @dispatch(str, str, list, object)
    def load_feature_dataframe(subject_id, session_id, feature_types, dataset):
        i = 0
        for feature_type in feature_types:
            feature_df = DataFrameLoader.load_epoched_dataframe(subject_id, session_id, feature_type, dataset)
            
            if i == 0:
                final_features = feature_df
            else:
                final_features = pd.merge(final_features, feature_df, how="inner", on=["epoch_timestamp"])
            
            i += 1
        return final_features
    
    @staticmethod
    @dispatch(str, list, object)
    def load_feature_dataframe(subject_id, feature_types, dataset): 
        session_ids = BuiltService.get_built_sleepsession_ids(subject_id, feature_types[0], dataset)

        feature_shape = DataFrameLoader.__get_dataframe_shape(subject_id, feature_types, dataset)
        
        stacked_feature_df = pd.DataFrame(np.zeros(feature_shape))
        
        current_height = 0
        for session_id in session_ids:
            
            try:
                feature_df = DataFrameLoader.load_feature_dataframe(subject_id, session_id, feature_types, dataset)
                columns = feature_df.columns
                feature_height = feature_df.shape[0]
                
                stacked_feature_df.iloc[current_height:(current_height + feature_height),:] = feature_df
                
                current_height += feature_height
            except:
                ExceptionLogger.append_exception(subject_id, session_id, str(feature_types), dataset.name, sys.exc_info()[0])
                print("Error: ", sys.exc_info()[0], " loading from DataFrameLoader for  " + str(subject_id) + ", session " + str(session_id))
        
        stacked_feature_df.columns = columns
        return stacked_feature_df
    
    @staticmethod
    @dispatch(list, object)
    def load_feature_dataframe(feature_types, dataset):
        
        subject_ids = BuiltService.get_built_subject_ids(feature_types[0], dataset)

        feature_shape = DataFrameLoader.__get_dataframe_shape(feature_types, dataset)
        
        stacked_feature_df = pd.DataFrame(np.zeros(feature_shape))
        
        current_height = 0
        for subject_id in subject_ids:
            
            try:
                feature_df = DataFrameLoader.load_feature_dataframe(subject_id, feature_types, dataset)
                columns = feature_df.columns
                feature_height = feature_df.shape[0]
                
                stacked_feature_df.iloc[current_height:(current_height + feature_height),:] = feature_df
                
                current_height += feature_height
            except:
                ExceptionLogger.append_exception(subject_id, 'N/A', str(feature_types), dataset.name, sys.exc_info()[0])
                print("Error: ", sys.exc_info()[0], " loading from DataFrameLoader for  " + str(subject_id))
                
        stacked_feature_df.columns = columns
        return stacked_feature_df
    
    @staticmethod
    @dispatch(list,list)
    def load_feature_dataframe(feature_types, datasets):

        feature_shape = DataFrameLoader.__get_dataframe_shape(feature_types, datasets)
        
        stacked_feature_df = pd.DataFrame(np.zeros(feature_shape))
        
        current_height = 0
        for dataset in datasets:
            
            feature_df = DataFrameLoader.load_feature_dataframe(feature_types, dataset)
            columns = feature_df.columns
            feature_height = feature_df.shape[0]
            
            stacked_feature_df.iloc[current_height:(current_height + feature_height),:] = feature_df
            
            current_height += feature_height
        
        stacked_feature_df.columns = columns
        return stacked_feature_df
            
    
    @staticmethod
    @dispatch(str, list, object)
    def __get_dataframe_shape(subject_id, feature_types, dataset):
        session_ids = BuiltService.get_built_sleepsession_ids(subject_id, feature_types[0], dataset)
        
        stacked_height = 0
        for i in range(len(session_ids)):
            
            try:
                feature_shape = DataFrameLoader.load_feature_dataframe(subject_id, session_ids[i], feature_types, dataset).to_numpy().shape
                    
                feature_height = feature_shape[0]
                feature_width = feature_shape[1]            
    
                # This has quite a bad runtime, might want to make it faster at some point
                stacked_height = stacked_height + feature_height
            except:
                pass
        
        return (stacked_height, feature_width)
    
    @staticmethod
    @dispatch(list, object)
    def __get_dataframe_shape(feature_types, dataset):

        subject_ids = BuiltService.get_built_subject_ids(feature_types[0], dataset)

        stacked_height = 0
        for i in range(len(subject_ids)):
            try:
                feature_shape = DataFrameLoader.__get_dataframe_shape(subject_ids[i], feature_types, dataset)
                feature_height = feature_shape[0]
                feature_width = feature_shape[1]

                stacked_height = stacked_height + feature_height
            except:
                pass
        
        return (stacked_height, feature_width)

    @staticmethod
    @dispatch(list, list)
    def __get_dataframe_shape(feature_types, datasets):
        
        i = 0
        for dataset in datasets:
            
            feature_shape = DataFrameLoader.__get_dataframe_shape(feature_types, dataset)
            feature_height = feature_shape[0]
            feature_width = feature_shape[1]
            
            if i == 0:
                stacked_height = feature_height
            else:
                # This has quite a bad runtime, might want to make it faster at some point
                stacked_height = stacked_height + feature_height
            i += 1
        
        return (stacked_height, feature_width)