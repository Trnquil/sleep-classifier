import os
import pandas as pd

from source.constants import Constants
from source.preprocessing.sleep_session_services.sleep_session_service import SleepSessionService
from source.analysis.setup.feature_type import FeatureType
from source.data_services.dataset import DataSet
from source.preprocessing.path_service import PathService

class BuiltService(object):
    
    def get_built_nightly_subject_and_sleepsession_ids(dataset):
        subject_to_session_dictionary = {}
        
        nightly_feature_path = PathService.get_nightly_feature_file_path(dataset)
        nightly_feature_dataframe = pd.read_csv(str(nightly_feature_path))
        nightly_feature_dataframe = nightly_feature_dataframe[['subject_id', 'session_id']]
        for index, row in nightly_feature_dataframe.iterrows():
            subject_id = row['subject_id']
            if(subject_id == "RESAMPLED"):
                continue
            session_id = row['session_id']
            if subject_id not in subject_to_session_dictionary.keys():
                subject_to_session_dictionary[subject_id] = []
                
            subject_to_session_dictionary[subject_id].append(session_id)
        
        return subject_to_session_dictionary
    
    # returns only subject and sleepsession ids for which there is data
    @staticmethod
    def get_built_subject_and_sleepsession_ids(feature_type, dataset):
        
        if(feature_type.name == FeatureType.nightly.name or feature_type.name in FeatureType.get_nightly_names()):
            return BuiltService.get_built_nightly_subject_and_sleepsession_ids(dataset)
        
        subject_to_session_dictionary = {}
        
        path = BuiltService.get_path(feature_type, dataset)

        
        if(dataset.name == DataSet.usi.name):
        
            for subject_id in Constants.SUBJECT_IDS:
                    if subject_id in os.listdir(path):
                        session_dirs = os.listdir(path.joinpath(subject_id))
                        for session_id in SleepSessionService.get_starttime_ordered_ids(subject_id, dataset):
                            if session_id in session_dirs:
                                if subject_id not in subject_to_session_dictionary.keys():
                                    subject_to_session_dictionary[subject_id] = []
                                    
                                subject_to_session_dictionary[subject_id].append(session_id)
            return subject_to_session_dictionary  
        
        
        elif(dataset.name == DataSet.mesa.name or dataset.name == DataSet.mss.name):
            for subject_id in os.listdir(path):
                    if '.' not in subject_id:
                        session_dirs = os.listdir(path.joinpath(subject_id))
                        for session_id in session_dirs:
                            if '.' not in session_id:
                                if subject_id not in subject_to_session_dictionary.keys():
                                    subject_to_session_dictionary[subject_id] = []
                                    
                                subject_to_session_dictionary[subject_id].append(session_id)
            return subject_to_session_dictionary 
            

    # returns only sleepsession ids for which there is data
    @staticmethod 
    def get_built_sleepsession_ids(subject_id, feature_type, dataset):
        return BuiltService.get_built_subject_and_sleepsession_ids(feature_type, dataset)[subject_id]    

    # returns only subject ids for which there is data
    @staticmethod 
    def get_built_subject_ids(feature_type, dataset):
        return list(BuiltService.get_built_subject_and_sleepsession_ids(feature_type, dataset).keys())
    
    # returns only subject ids for which there is data
    @staticmethod 
    def get_built_sleepsession_count(feature_type, dataset):
        count = 0
        for subject_id in BuiltService.get_built_subject_ids(feature_type, dataset):
            for session_id in BuiltService.get_built_sleepsession_ids(subject_id, feature_type, dataset):
                count += 1
        return count
    
    @staticmethod 
    def get_path(feature_type, dataset):
        # Getting the correct path for every featuretype and dataset
        # For nightly, we assume that the same subjects and sessions will have been built than for epoched
        
        # epoched features
        if (feature_type.name == FeatureType.epoched.name or feature_type.name in FeatureType.get_epoched_names()
        or feature_type.name == FeatureType.sleep_quality.name):
            if(dataset.name == DataSet.usi.name):
                path = Constants.EPOCHED_FILE_PATH.joinpath(Constants.USI_FOLDER_NAME)
            elif(dataset.name == DataSet.mesa.name):
                path = Constants.EPOCHED_FILE_PATH.joinpath(Constants.MESA_FOLDER_NAME)
            elif(dataset.name == DataSet.mss.name):
                path = Constants.EPOCHED_FILE_PATH.joinpath(Constants.MSS_FOLDER_NAME)
           
        # cropped features
        elif feature_type.name == FeatureType.cropped.name or feature_type.name in FeatureType.get_cropped_names():
            if(dataset.name == DataSet.usi.name):
                path = Constants.CROPPED_FILE_PATH.joinpath(Constants.USI_FOLDER_NAME)
            elif(dataset.name == DataSet.mss.name):
                path = Constants.CROPPED_FILE_PATH.joinpath(Constants.MSS_FOLDER_NAME)
            
        # clusters
        elif feature_type.name == FeatureType.cluster.name or feature_type.name == FeatureType.cluster_features.name:
            if(dataset.name == DataSet.usi.name):
                path = Constants.CLUSTERS_FILE_PATH.joinpath(Constants.USI_FOLDER_NAME)
            elif(dataset.name == DataSet.mesa.name):
                path = Constants.CLUSTERS_FILE_PATH.joinpath(Constants.MESA_FOLDER_NAME)
            elif(dataset.name == DataSet.mss.name):
                path = Constants.CLUSTERS_FILE_PATH.joinpath(Constants.MSS_FOLDER_NAME)
        return path
    
    