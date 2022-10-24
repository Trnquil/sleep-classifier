from source.preprocessing.path_service import PathService
import joblib
import numpy as np
import pandas as pd

from source.analysis.setup.feature_type import FeatureType


class ClusteringService(object):
    
    @staticmethod
    def get_predictions(features):
        
        classifier = joblib.load(PathService.get_model_path())
        
        class_predictions = classifier.predict(features)
        
        return class_predictions
    
    @staticmethod
    def load(subject_id, session_id):
        cluster_feature_path = PathService.get_epoched_file_path(subject_id, session_id, FeatureType.epoched_cluster)
        feature = pd.read_csv(str(cluster_feature_path)).values
        return feature