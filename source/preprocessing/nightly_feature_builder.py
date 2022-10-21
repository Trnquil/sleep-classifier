import sys
# caution: path[0] is reserved for script path (or '' in REPL)
# We are inserting the sleepclassifiers as a module 
sys.path.insert(1, '../..')

from source.constants import Constants
from source.preprocessing.clustering.clustering_final_feature_service import ClusteringFinalFeatureService
from source.preprocessing.path_service import PathService
from source.analysis.setup.sleep_session_service import SleepSessionService

import pandas as pd




class NightlyFeatureBuilder(object):

    @staticmethod
    def build(subject_id, session_id):

        if Constants.VERBOSE:
            print("Building nightly features...")
        
        cluster_final_feature_dict = ClusteringFinalFeatureService.build(subject_id, session_id)
        
        # For now we only put some cluster features in here
        sleepquality = SleepSessionService.load_sleepquality(subject_id, session_id)
        sleepquality_dict = {'sleep_quality': [sleepquality]}
        final_features = cluster_final_feature_dict | sleepquality_dict
        
        final_features_dataframe = pd.DataFrame(final_features)
        
        # Writing all features to their files
        final_features_path = NightlyFeatureBuilder.get_path(subject_id, session_id)
        final_features_dataframe.to_csv(final_features_path, index=False)
        
    @staticmethod
    def load(subject_id, session_id):
        nightly_feature_path = NightlyFeatureBuilder().get_path(subject_id, session_id)
        nightly_feature_dataframe = pd.read_csv(str(nightly_feature_path))
        return nightly_feature_dataframe
    
    @staticmethod
    def get_path(subject_id, session_id):
        return PathService.get_final_folder_path(subject_id, session_id) + '/nightly_features.csv'
            
                                     

        
