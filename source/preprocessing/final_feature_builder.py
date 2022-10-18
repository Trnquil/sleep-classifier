import sys
# caution: path[0] is reserved for script path (or '' in REPL)
# We are inserting the sleepclassifiers as a module 
sys.path.insert(1, '../..')

from source.constants import Constants
from source.preprocessing.clustering.clustering_final_feature_service import ClusteringFinalFeatureService
from source.preprocessing.path_service import PathService
from source.analysis.setup.sleep_session_service import SleepSessionService

import pandas as pd




class FinalFeatureBuilder(object):

    @staticmethod
    def build(subject_id, session_id):

        if Constants.VERBOSE:
            print("Building final features...")
        
        cluster_final_feature_dict = ClusteringFinalFeatureService.build(subject_id, session_id)
        
        # For now we only put some cluster features in here
        sleepquality = SleepSessionService.load_sleepquality(subject_id, session_id)
        sleepquality_dict = {'sleep_quality': [sleepquality]}
        final_features = cluster_final_feature_dict | sleepquality_dict
        
        final_features_dataframe = pd.DataFrame(final_features)
        
        # Writing all features to their files
        final_features_path = PathService.get_final_folder_path(subject_id, session_id) + '/final_features.csv'
        final_features_dataframe.to_csv(final_features_path)
            
            
                                     

        
