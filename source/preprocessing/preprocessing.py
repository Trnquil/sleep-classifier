import sys
sys.path.insert(1, '../..')

from source.preprocessing.built_service import BuiltService
from source.constants import Constants
from source.preprocessing.activity_count.activity_count_service import ActivityCountService
from source.preprocessing.usi_epoched_feature_builder import UsiEpochedFeatureBuilder
from source.preprocessing.mss_epoched_feature_builder import MssEpochedFeatureBuilder
from source.preprocessing.nightly.nightly_feature_builder import NightlyFeatureBuilder
from source.preprocessing.raw_data_processor import RawDataProcessor
from source.preprocessing.mss_raw_data_processor import MssRawDataProcessor
from source.preprocessing.time.circadian_service import CircadianService
from source.preprocessing.clustering.cluster_feature_service import ClusterFeatureService
from source.preprocessing.clustering.cluster_builder import ClusterBuilder
from source.analysis.setup.feature_type import FeatureType
from source.data_services.dataset import DataSet
from source.mesa.mesa_data_service import MesaDataService
from source.mesa.mesa_feature_builder import MesaFeatureBuilder
from source.preprocessing.clustering.cluster_feature_builder import ClusterFeatureBuilder
from tqdm import tqdm


class Preprocessing(object):
    
    @staticmethod
    def build_usi_cropped():
        subject_ids = Constants.SUBJECT_IDS
        with tqdm(subject_ids, leave=True, unit='subject', colour='green') as t:
            t.set_description("Building USI Cropped")
            for subject in t:
                RawDataProcessor.crop_all(str(subject))
        
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
        clustering_model = ClusterFeatureService.get_fitted_model()
        # Only building features for subjects and sleepsession for which folders exist
        subject_sleepsession_dictionary = BuiltService.get_built_subject_and_sleepsession_ids(FeatureType.epoched, DataSet.usi)
        
        with tqdm(subject_sleepsession_dictionary.keys(), unit='subject', colour='green') as t:
            t.set_description("Building USI Clusters")
            for subject in t:
                for session in subject_sleepsession_dictionary[subject]:
                    ClusterBuilder.build(subject, session, DataSet.usi, clustering_model)
                
        subject_sleepsession_dictionary = BuiltService.get_built_subject_and_sleepsession_ids(FeatureType.epoched, DataSet.mesa)
        with tqdm(subject_sleepsession_dictionary.keys(), unit='subject', colour='green') as t:
            for subject in t:
                t.set_description("Building MESA Clusters")
                for session in subject_sleepsession_dictionary[subject]:
                    ClusterBuilder.build(subject, session, DataSet.mesa, clustering_model)
                    
        # subject_sleepsession_dictionary = BuiltService.get_built_subject_and_sleepsession_ids(FeatureType.epoched, DataSet.mss)
        # with tqdm(subject_sleepsession_dictionary.keys(), unit='subject', colour='green') as t:
        #     for subject in t:
        #         t.set_description("Building MSS Clusters")
        #         for session in subject_sleepsession_dictionary[subject]:
        #             ClusterBuilder.build(subject, session, DataSet.mss, clustering_model)
                
    @staticmethod   
    def build_nightly_usi():
        NightlyFeatureBuilder.build(DataSet.usi)
    
    @staticmethod   
    def build_nightly_mss():
        NightlyFeatureBuilder.build(DataSet.mss)
    
    
# for subject_id in subject_ids:
    # DataPlotBuilder.make_data_demo(subject_id, False)
