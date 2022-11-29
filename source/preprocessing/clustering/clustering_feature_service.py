from source.preprocessing.path_service import PathService
import joblib
import numpy as np
import pandas as pd

from source.analysis.setup.feature_type import FeatureType
from source.data_services.data_service import DataService
from sklearn.cluster import KMeans
from source.data_services.dataset import DataSet
from source.data_services.data_loader import DataLoader
from source.data_services.data_frame_loader import DataFrameLoader




class ClusteringFeatureService(object):
    @staticmethod
    def get_predictions_from_imported_model(features):
        
        classifier = joblib.load(PathService.get_model_path())
        
        class_predictions = classifier.predict(features)
        
        return class_predictions
    
    @staticmethod
    def get_fitted_model(dataset):
        
        features_df = DataFrameLoader.load_feature_dataframe([FeatureType.cluster_features], dataset)
        features = features_df.drop(columns=['epoch_timestamp']).to_numpy().squeeze()
        classifier=KMeans(n_clusters=6, random_state=0)

        # We are now fitting our features to the cluster
        classifier.fit(features)
        
        return classifier
    

            

        