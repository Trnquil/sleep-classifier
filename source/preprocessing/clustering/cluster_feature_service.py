from source.preprocessing.path_service import PathService
import joblib

from source.analysis.setup.feature_type import FeatureType
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from source.data_services.data_frame_loader import DataFrameLoader
from source.runner_parameters import RunnerParameters
from source.analysis.setup.clustering_algorithm import ClusteringAlgorithm
from source.preprocessing.sleep_wake import SleepWake
from source.data_services.dataset import DataSet


class ClusterFeatureService(object):
    @staticmethod
    def get_predictions_from_imported_model(features):
        
        classifier = joblib.load(PathService.get_model_path())
        
        class_predictions = classifier.predict(features)
        
        return class_predictions
    
    @staticmethod
    def get_fitted_model(clustering_algorithm, sleep_wake):
        
        # Only Usi Dataset has wake and selfreported sleep information
        if(sleep_wake.name == SleepWake.wake.name or sleep_wake.name == SleepWake.selfreported_sleep.name):   
            features_df = DataFrameLoader.load_feature_dataframe([FeatureType.cluster_features], sleep_wake, [DataSet.usi])
        else:
            features_df = DataFrameLoader.load_feature_dataframe([FeatureType.cluster_features], sleep_wake, RunnerParameters.CLUSTERING_DATASETS)
        features = features_df.drop(columns=['epoch_timestamp', 'subject_id', 'session_id']).dropna().to_numpy().squeeze()
        
        if clustering_algorithm.name == ClusteringAlgorithm.GMM.name:
            classifier = GaussianMixture(n_components=RunnerParameters.NUMBER_OF_CLUSTERS)
            
        elif clustering_algorithm.name == ClusteringAlgorithm.KMeans.name:
            classifier = KMeans(n_clusters=RunnerParameters.NUMBER_OF_CLUSTERS, random_state=0)


        # We are now fitting our features to the cluster
        classifier.fit(features)
        
        return classifier
    

            

        