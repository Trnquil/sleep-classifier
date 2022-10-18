from source.preprocessing.path_service import PathService
import joblib
import numpy as np


class ClusteringService(object):
    
    @staticmethod
    def get_model_path():
        return PathService.get_models_path() + "/kmeans.pkl"
    
    @staticmethod
    def get_predictions(features):
        
        classifier = joblib.load(ClusteringService.get_model_path())
        
        class_predictions = classifier.predict(features)
        
        return class_predictions
    
    @staticmethod
    def get_path(subject_id, session_id):
        return PathService.get_feature_folder_path(subject_id, session_id) + '/clusters.out'
    
    @staticmethod
    def write(subject_id, session_id, feature):
        cluster_feature_path = ClusteringService.get_path(subject_id, session_id)
        np.savetxt(cluster_feature_path, feature, fmt='%f')