import pandas as pd
from source.preprocessing.path_service import PathService


class NightlyFeatureService(object):

    @staticmethod
    def load(subject_id, session_id):
        nightly_feature_path = NightlyFeatureService.get_path(subject_id, session_id)
        nightly_feature_dataframe = pd.read_csv(str(nightly_feature_path))
        return nightly_feature_dataframe
    
    @staticmethod
    def get_path():
        return PathService.get_nightly_folder_path() + '/nightly_features.csv'
            