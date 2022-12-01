from source.preprocessing.path_service import PathService
import joblib

from source.analysis.setup.feature_type import FeatureType
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from source.data_services.data_frame_loader import DataFrameLoader
from source.runner_parameters import RunnerParameters
from source.analysis.setup.clustering_algorithm import ClusteringAlgorithm



class ClusterFeatureService(object):
    @staticmethod
    def get_predictions_from_imported_model(features):
        
        classifier = joblib.load(PathService.get_model_path())
        
        class_predictions = classifier.predict(features)
        
        return class_predictions
    
    @staticmethod
    def get_fitted_model():
        
        features_df = DataFrameLoader.load_feature_dataframe([FeatureType.cluster_features], RunnerParameters.CLUSTERING_DATASETS)
        features = features_df.drop(columns=['epoch_timestamp']).to_numpy().squeeze()
        
        if RunnerParameters.CLUSTERING_ALGO.name == ClusteringAlgorithm.GMM.name:
            classifier = GaussianMixture(n_components=RunnerParameters.NUMBER_OF_CLUSTERS)
            
        elif RunnerParameters.CLUSTERING_ALGO.name == ClusteringAlgorithm.KMeans.name:
            classifier = KMeans(n_clusters=RunnerParameters.NUMBER_OF_CLUSTERS, random_state=0)


        # We are now fitting our features to the cluster
        classifier.fit(features)
        
        return classifier
    

            

        