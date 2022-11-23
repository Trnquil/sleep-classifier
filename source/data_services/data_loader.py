from source.analysis.setup.feature_type import FeatureType
from source.preprocessing.path_service import PathService
from source.constants import Constants
from source import utils
from source.preprocessing.collection import Collection
from source.preprocessing.built_service import BuiltService
from source.data_services.dataset import DataSet


import numpy as np
import pandas as pd
import os

from multipledispatch import dispatch

class DataLoader(object):
    
    @staticmethod
    def load_raw(subject_id, feature_type):
        
        if(feature_type.name == FeatureType.raw_ibi.name):
            return DataLoader.load_raw_ibi(subject_id)
        
        raw_paths = PathService.get_raw_file_paths(subject_id, feature_type)
        
        final_feature_shape = DataLoader.load_raw_shape(subject_id, feature_type)
        final_feature_array = np.zeros(final_feature_shape)

        current_height = 0
        for raw_path in raw_paths:
            if os.path.getsize(str(raw_path))==0:
                continue
            feature_array = pd.read_csv(str(raw_path))
            
            #unix start time of the data
            start_time = feature_array.columns.values.tolist()
            start_time = float(start_time[0])
            
            #Let us center the time so that time values aren't too big
            start_time = start_time - Constants.TIME_CENTER
            
            if(feature_array.ndim == 1):
                np.expand_dims(feature_array, axis=1)
            
            feature_array = feature_array.to_numpy()
            
            #data value frequency in Hertz
            data_frequency = feature_array[0][0]
    
            feature_array = feature_array[1:][:]
            feature_height = feature_array.shape[0]
            
            # We need to normalize acceleration data
            if(feature_type == FeatureType.raw_acc):
                feature_array = feature_array/64
            
            timestamps = [(start_time + i*(1/data_frequency)) for i in range(len(feature_array))]
            timestamps = np.expand_dims(timestamps, axis=1)
            feature_array = np.concatenate([timestamps, feature_array], axis=1)
            feature_array = utils.remove_repeats(feature_array)
            
            final_feature_array[current_height:(current_height + feature_height)][:] = feature_array
            current_height += feature_height
            
        return Collection(subject_id=subject_id, data=final_feature_array, data_frequency=data_frequency)
    
    
    @staticmethod
    def load_raw_ibi(subject_id):
        raw_paths = PathService.get_raw_file_paths(subject_id, FeatureType.raw_ibi)
        
        final_feature_shape = DataLoader.load_raw_shape(subject_id, FeatureType.raw_ibi)
        final_feature_array = np.zeros(final_feature_shape)

        current_height = 0
        for raw_path in raw_paths:
            if os.path.getsize(str(raw_path))==0:
                continue
            feature_array = pd.read_csv(str(raw_path))
            
            #unix start time of the data
            start_time = feature_array.columns.values.tolist()
            start_time = float(start_time[0])
            
            #Let us center the time so that time values aren't too big
            start_time = start_time - Constants.TIME_CENTER
            
            feature_array = feature_array.to_numpy()
    
            feature_height = feature_array.shape[0]
            
            #We bring the times into our centered time format
            feature_array[:,0] = feature_array[:,0] + start_time
            
            final_feature_array[current_height:(current_height + feature_height)][:] = feature_array
            current_height += feature_height
            
        return Collection(subject_id=subject_id, data=final_feature_array, data_frequency=0)
        
    @staticmethod
    def load_raw_shape(subject_id, feature_type):
        raw_paths = PathService.get_raw_file_paths(subject_id, feature_type)
        feature_height = 0

        for raw_path in raw_paths:
            if os.path.getsize(str(raw_path))==0:
                continue
            feature_array = pd.read_csv(str(raw_path))
            
            if(feature_array.ndim == 1):
                np.expand_dims(feature_array, axis=1)
            
            feature_array = feature_array.to_numpy()
            
            # We subtract one because of the data frequency row, except in ibi
            feature_height += feature_array.shape[0] + (0 if feature_type.name == FeatureType.raw_ibi.name else -1)
            
            # We add one because of the time column, except in ibi
            feature_width = feature_array.shape[1] + (0 if feature_type.name == FeatureType.raw_ibi.name else 1)
            
        return (feature_height, feature_width)
    
    @staticmethod
    def load_cropped(subject_id, session_id, feature_type):
        file_path = PathService.get_cropped_file_path(subject_id, session_id, feature_type)
        values = pd.read_csv(str(file_path), delimiter=" ").values
        return Collection(subject_id=subject_id, data=values, data_frequency=0)
    
    @staticmethod
    def load_epoched(subject_id, session_id, feature_type, dataset):
        
        feature_path = PathService.get_epoched_file_path(subject_id, session_id, feature_type, dataset)
        feature_dataframe = pd.read_csv(str(feature_path))

        return feature_dataframe
    
    @staticmethod
    def load_epoched_columns(feature_type, dataset):
        subject_sleepsession_dictionary = BuiltService.get_built_subject_and_sleepsession_ids(FeatureType.epoched, dataset)
        subject_id = list(subject_sleepsession_dictionary.keys())[0]
        session_id = subject_sleepsession_dictionary[subject_id][0]
        feature_dataframe = DataLoader.load_epoched(subject_id, session_id, feature_type, dataset)
        return feature_dataframe.columns
    
    @staticmethod
    @dispatch()
    def load_nightly():
        nightly_feature_path = PathService.get_nightly_file_path()
        nightly_feature_dataframe = pd.read_csv(str(nightly_feature_path))
        return nightly_feature_dataframe
    
    @staticmethod
    @dispatch(str, str, object)
    def load_nightly(subject_id, session_id, feature_type):
        nightly_feature_path = PathService.get_nightly_file_path()
        nightly_feature_dataframe = pd.read_csv(str(nightly_feature_path))
    
        
        # making sure we only work with data from the requested user and session
        nightly_feature_dataframe = nightly_feature_dataframe[nightly_feature_dataframe['subject_id'].eq(subject_id)]
        nightly_feature_dataframe = nightly_feature_dataframe[nightly_feature_dataframe['session_id'].eq(session_id)]
        
        if(feature_type.name == FeatureType.nightly_cluster.name):
            nightly_feature_dataframe = nightly_feature_dataframe.filter(regex=("c_.*"))
        elif(feature_type.name == FeatureType.nightly_count.name):
            nightly_feature_dataframe = nightly_feature_dataframe.filter(regex=("count_.*"))
        elif(feature_type.name == FeatureType.nightly_ibi.name):
            nightly_feature_dataframe = nightly_feature_dataframe.filter(regex=("ibi_.*"))
        elif(feature_type.name == FeatureType.nightly_hr.name):
            nightly_feature_dataframe = nightly_feature_dataframe.filter(regex=("hr_.*"))
        elif(feature_type.name == FeatureType.nightly_sleep_quality.name):
            nightly_feature_dataframe = nightly_feature_dataframe.filter(regex=("sleep_quality"))
        else:
            raise Exception("FeatureType unknown to DataLoader")
        
        return nightly_feature_dataframe
    
    @staticmethod
    def load_common_epoched_timestamps(subject_id, session_id, feature_types, dataset):
        i = 0
        for feature_type in feature_types:
            timestamp_df = DataLoader.load_epoched(subject_id, session_id, feature_type, dataset)['epoch_timestamp']
            if i == 0:
                common_timestamps = timestamp_df
            else:
                common_timestamps[common_timestamps.isin(timestamp_df)]
            i += 1
        return common_timestamps