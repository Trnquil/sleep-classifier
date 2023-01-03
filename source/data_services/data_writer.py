from source.preprocessing.path_service import PathService

import numpy as np

class DataWriter(object):
    
    @staticmethod
    def write_cropped(collection, session_id, feature_type, sleep_wake, dataset):
        PathService.create_cropped_folder_path(collection.subject_id, session_id, sleep_wake, dataset)
        output_path = PathService.get_cropped_file_path(collection.subject_id, session_id, feature_type, sleep_wake, dataset)
        np.savetxt(output_path, collection.data, fmt='%f')
    
    @staticmethod
    def write_epoched(epoched_feature, subject_id, session_id, feature_type, sleep_wake, dataset):
        PathService.create_epoched_folder_path(subject_id, session_id, sleep_wake, dataset)
        feature_path = PathService.get_epoched_file_path(subject_id, session_id, feature_type, sleep_wake, dataset)
        epoched_feature.to_csv(feature_path, index=False)
        
    @staticmethod
    def write_cluster(cluster_feature, subject_id, session_id, feature_type, sleep_wake, dataset):
        PathService.create_clusters_folder_path(subject_id, session_id, sleep_wake, dataset)
        feature_path = PathService.get_clusters_file_path(subject_id, session_id, feature_type, sleep_wake, dataset)
        cluster_feature.to_csv(feature_path, index=False)
    
    @staticmethod
    def write_nightly(nightly_dataframe, feature_type, dataset):
        PathService.create_nightly_folder_path(dataset)
        nightly_feature_path = PathService.get_nightly_file_path(feature_type, dataset)
        nightly_dataframe.to_csv(nightly_feature_path, index=False)