from source.analysis.setup.feature_type import FeatureType
from source.data_services.data_frame_loader import DataFrameLoader
from source.data_services.dataset import DataSet
from source.preprocessing.built_service import BuiltService
from source.data_services.data_writer import DataWriter
from source.preprocessing.path_service import PathService
from source.constants import Constants

import numpy as np

class ClusterFeatureBuilder(object):
    cluster_feature_types = [FeatureType.epoched_hr, FeatureType.epoched_count]
    
    @staticmethod
    def build(dataset):
     
        subject_sleepsession_dictionary = BuiltService.get_built_subject_and_sleepsession_ids(FeatureType.epoched, dataset)
        
        for subject_id in subject_sleepsession_dictionary.keys():
            
            if Constants.VERBOSE:
                print("Building cluster features for subject " + str(subject_id) + "...")
            subject_df = DataFrameLoader.load_feature_dataframe(subject_id, ClusterFeatureBuilder.cluster_feature_types, dataset)
            subject_mean = np.mean(subject_df.iloc[:,1:], axis=0)
            subject_std = np.std(subject_df.iloc[:,1:], axis=0)
            
            for session_id in subject_sleepsession_dictionary[subject_id]:
                
                session_df = DataFrameLoader.load_feature_dataframe(subject_id, session_id, ClusterFeatureBuilder.cluster_feature_types, dataset)
                normalized_session_df = session_df
                normalized_session_df.iloc[:,1:] = (normalized_session_df.iloc[:,1:] - subject_mean)/subject_std
                normalized_session_df[['count']] = normalized_session_df[['count']] + (subject_mean['count']/subject_std['count'])
               
                PathService.create_clusters_folder_path(subject_id, session_id, dataset)
                DataWriter.write_cluster(normalized_session_df, subject_id, session_id, FeatureType.cluster_features, dataset)
        

    