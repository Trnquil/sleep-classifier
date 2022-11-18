from source.preprocessing.path_service import PathService
import joblib
import numpy as np
import pandas as pd

from source.analysis.setup.feature_type import FeatureType
from source.data_services.data_service import DataService
from sklearn.cluster import KMeans
from source.data_services.dataset import DataSet




class ClusteringFeatureService(object):
    
    @staticmethod
    def get_predictions_from_imported_model(features):
        
        classifier = joblib.load(PathService.get_model_path())
        
        class_predictions = classifier.predict(features)
        
        return class_predictions
    
    @staticmethod
    def get_fitted_model():
        
        features = DataService.load_feature_raw(FeatureType.epoched, DataSet.usi)[:,1:].squeeze()
        classifier=KMeans(n_clusters=6, random_state=0)
        

        # We are now fitting our features to the cluster
        classifier.fit(features)
        
        return classifier
    
    @staticmethod
    def load(subject_id, session_id):
        cluster_feature_path = PathService.get_epoched_file_path(subject_id, session_id, FeatureType.epoched_cluster)
        feature = pd.read_csv(str(cluster_feature_path)).values
        return feature