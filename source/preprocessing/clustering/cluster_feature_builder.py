from source.analysis.setup.feature_type import FeatureType
from source.data_services.data_frame_loader import DataFrameLoader
from source.data_services.dataset import DataSet
from source.preprocessing.built_service import BuiltService
from source.data_services.data_writer import DataWriter
from source.preprocessing.path_service import PathService
from source.constants import Constants
from source.runner_parameters import RunnerParameters

import numpy as np
from tqdm import tqdm

class ClusterFeatureBuilder(object):
    
    @staticmethod
    def build(dataset):
        
        if not RunnerParameters.CLUSTERING_PER_SUBJECT_NORMALIZATION:
            subject_df = DataFrameLoader.load_feature_dataframe(RunnerParameters.CLUSTERING_FEATURES, dataset)
            overall_mean = np.mean(subject_df.iloc[:,1:], axis=0)
            overall_std = np.std(subject_df.iloc[:,1:], axis=0)
        
        subject_sleepsession_dictionary = BuiltService.get_built_subject_and_sleepsession_ids(FeatureType.epoched, dataset)
        
        with tqdm(subject_sleepsession_dictionary.keys(), unit='subject', colour='green') as t:
            for subject_id in t:
                t.set_description("Building " + dataset.name.upper() + " Cluster Features")
                
                subject_df = DataFrameLoader.load_feature_dataframe(subject_id, RunnerParameters.CLUSTERING_FEATURES, dataset)
                subject_mean = np.mean(subject_df.iloc[:,1:], axis=0)
                subject_std = np.std(subject_df.iloc[:,1:], axis=0)
                
                for session_id in subject_sleepsession_dictionary[subject_id]:
                    
                    session_df = DataFrameLoader.load_feature_dataframe(subject_id, session_id, RunnerParameters.CLUSTERING_FEATURES, dataset)
                    normalized_session_df = session_df
                    
                    if RunnerParameters.CLUSTERING_PER_SUBJECT_NORMALIZATION:
                        normalized_session_df.iloc[:,1:] = (normalized_session_df.iloc[:,1:] - subject_mean)/subject_std
                        normalized_session_df[['count']] = normalized_session_df[['count']] + (subject_mean['count']/subject_std['count'])
                    else:
                        normalized_session_df.iloc[:,1:] = (normalized_session_df.iloc[:,1:] - overall_mean)/overall_std
                        normalized_session_df[['count']] = normalized_session_df[['count']] + (overall_mean['count']/overall_std['count'])
                   
                    PathService.create_clusters_folder_path(subject_id, session_id, dataset)
                    DataWriter.write_cluster(normalized_session_df, subject_id, session_id, FeatureType.cluster_features, dataset)
        

    