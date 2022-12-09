from source.preprocessing.path_service import PathService
from source.analysis.setup.feature_type import FeatureType

import numpy as np


class DataWriter(object):
    
    @staticmethod
    def write_cropped(collection, session_id, feature_type, dataset):
        output_path = PathService.get_cropped_file_path(collection.subject_id, session_id, feature_type, dataset)
        np.savetxt(output_path, collection.data, fmt='%f')
    
    @staticmethod
    def write_epoched(epoched_feature, subject_id, session_id, feature_type, dataset):
        feature_path = PathService.get_epoched_file_path(subject_id, session_id, feature_type, dataset)
        epoched_feature.to_csv(feature_path, index=False)
        
    @staticmethod
    def write_cluster(cluster_feature, subject_id, session_id, feature_type, dataset):
        feature_path = PathService.get_clusters_file_path(subject_id, session_id, feature_type, dataset)
        cluster_feature.to_csv(feature_path, index=False)

    
    @staticmethod
    def write_nightly(nightly_dataframe, feature_type):
        nightly_feature_path = PathService.get_nightly_file_path(feature_type)
        nightly_dataframe.to_csv(nightly_feature_path, index=False)