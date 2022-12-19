

from source.preprocessing.built_service import BuiltService
from source.constants import Constants
from source.preprocessing.activity_count.activity_count_service import ActivityCountService
from source.preprocessing.usi_epoched_feature_builder import UsiEpochedFeatureBuilder
from source.preprocessing.mss_epoched_feature_builder import MssEpochedFeatureBuilder
from source.preprocessing.nightly.nightly_feature_builder import NightlyFeatureBuilder
from source.preprocessing.usi_raw_data_processor import UsiRawDataProcessor
from source.preprocessing.mss_raw_data_processor import MssRawDataProcessor
from source.preprocessing.time.circadian_service import CircadianService
from source.preprocessing.clustering.cluster_feature_service import ClusterFeatureService
from source.preprocessing.clustering.cluster_builder import ClusterBuilder
from source.analysis.setup.feature_type import FeatureType
from source.data_services.dataset import DataSet
from source.mesa.mesa_data_service import MesaDataService
from source.mesa.mesa_feature_builder import MesaFeatureBuilder
from source.preprocessing.clustering.cluster_feature_builder import ClusterFeatureBuilder
from source.runner_parameters import RunnerParameters
from source.analysis.setup.clustering_algorithm import ClusteringAlgorithm

from tqdm import tqdm

from GEMINI.gemini_service import GeminiService


class Preprocessing(object):
    
    @staticmethod
    def build_usi_cropped():
        subject_ids = Constants.SUBJECT_IDS
        with tqdm(subject_ids, leave=True, unit='subject', colour='green') as t:
            t.set_description("Building USI Cropped")
            for subject in t:
                UsiRawDataProcessor.crop_all(str(subject))
        
            if Constants.INCLUDE_CIRCADIAN:
                ActivityCountService.build_activity_counts()  # This uses MATLAB, but has been replaced with a python implementation
                CircadianService.build_circadian_model()      # Both of the circadian lines require MATLAB to run
                CircadianService.build_circadian_mesa()       # INCLUDE_CIRCADIAN = False by default because most people don't have MATLAB
                
    @staticmethod
    def build_mss_cropped():
        subject_ids = MssRawDataProcessor.get_user_ids()
        with tqdm(subject_ids, leave=True, unit='subject', colour='green') as t:
            t.set_description("Building MSS Cropped")
            for subject in t:
                MssRawDataProcessor.crop_all(str(subject))
        
        
    @staticmethod
    def build_usi_epoched():
        # Building and saving a GEMINI Model
        if(RunnerParameters.GEMINI_TRAIN or (not GeminiService.model_exists())):
            print("Building GEMINI Model... ")
            GeminiService.build_model()
            
        # Only building features for subjects and sleepsession for which folders exist
        subject_ids = BuiltService.get_built_subject_ids(FeatureType.cropped, DataSet.usi)
        with tqdm(subject_ids, leave = True, unit='subject', colour='green') as t:
            for subject_id in t:
                t.set_description("Building USI Epoched")
                sleepsessions = BuiltService.get_built_sleepsession_ids(subject_id, FeatureType.cropped, DataSet.usi)
                for session_id in sleepsessions:
                    UsiEpochedFeatureBuilder.build(subject_id, session_id)
            
    @staticmethod
    def build_mesa_epoched():
        subject_ids = MesaDataService.get_all_subject_ids()
        with tqdm(subject_ids, leave = True, colour ='green', unit='subject') as t:
            for subject_id in t:
                t.set_description("Building MESA Epoched")
                MesaFeatureBuilder.build(subject_id)
                
    @staticmethod
    def build_mss_epoched():
        # Only building features for subjects and sleepsession for which folders exist
        subject_ids = BuiltService.get_built_subject_ids(FeatureType.cropped, DataSet.mss)
        with tqdm(subject_ids, leave = True, unit='subject', colour='green') as t:
            for subject_id in t:
                t.set_description("Building MSS Epoched")
                sleepsessions = BuiltService.get_built_sleepsession_ids(subject_id, FeatureType.cropped, DataSet.mss)
                for session_id in sleepsessions:
                    MssEpochedFeatureBuilder.build(subject_id, session_id)          

            
    @staticmethod    
    def build_cluster_features_usi():
            ClusterFeatureBuilder.build(DataSet.usi)
    
    @staticmethod   
    def build_cluster_features_mesa():
            ClusterFeatureBuilder.build(DataSet.mesa)
    
    @staticmethod   
    def build_cluster_features_mss():
            ClusterFeatureBuilder.build(DataSet.mss)
    
    @staticmethod   
    def build_clusters():
        clustering_model_kmeans = ClusterFeatureService.get_fitted_model(ClusteringAlgorithm.KMeans)
        clustering_model_gmm = ClusterFeatureService.get_fitted_model(ClusteringAlgorithm.GMM)
        
        # Only building features for subjects and sleepsession for which folders exist
        subject_sleepsession_dictionary = BuiltService.get_built_subject_and_sleepsession_ids(FeatureType.epoched, DataSet.usi)
        
        with tqdm(subject_sleepsession_dictionary.keys(), unit='subject', colour='green') as t:
            t.set_description("Building USI Clusters")
            for subject in t:
                for session in subject_sleepsession_dictionary[subject]:
                    ClusterBuilder.build(subject, session, FeatureType.cluster_kmeans, DataSet.usi, clustering_model_kmeans)
                    ClusterBuilder.build(subject, session, FeatureType.cluster_gmm, DataSet.usi, clustering_model_gmm)
                
        subject_sleepsession_dictionary = BuiltService.get_built_subject_and_sleepsession_ids(FeatureType.epoched, DataSet.mesa)
        with tqdm(subject_sleepsession_dictionary.keys(), unit='subject', colour='green') as t:
            for subject in t:
                t.set_description("Building MESA Clusters")
                for session in subject_sleepsession_dictionary[subject]:
                    ClusterBuilder.build(subject, session, FeatureType.cluster_kmeans, DataSet.mesa, clustering_model_kmeans)
                    ClusterBuilder.build(subject, session, FeatureType.cluster_gmm, DataSet.mesa, clustering_model_gmm)
                    
        subject_sleepsession_dictionary = BuiltService.get_built_subject_and_sleepsession_ids(FeatureType.epoched, DataSet.mss)
        with tqdm(subject_sleepsession_dictionary.keys(), unit='subject', colour='green') as t:
            for subject in t:
                t.set_description("Building MSS Clusters")
                for session in subject_sleepsession_dictionary[subject]:
                    ClusterBuilder.build(subject, session, FeatureType.cluster_kmeans, DataSet.mss, clustering_model_kmeans)
                    ClusterBuilder.build(subject, session, FeatureType.cluster_gmm, DataSet.mss, clustering_model_gmm)
                
    @staticmethod   
    def build_nightly_usi():
        NightlyFeatureBuilder.build(DataSet.usi)
    
    @staticmethod   
    def build_nightly_mss():
        NightlyFeatureBuilder.build(DataSet.mss)
    
    
# for subject_id in subject_ids:
    # DataPlotBuilder.make_data_demo(subject_id, False)
