from source.preprocessing.path_service import PathService
from source.analysis.setup.feature_type import FeatureType

import numpy as np


class DataWriter(object):
    
    @staticmethod
    def write_cropped(collection, session_id, feature_type):
        output_path = PathService.get_cropped_file_path(collection.subject_id, session_id, feature_type)
        np.savetxt(output_path, collection.data, fmt='%f')
    
    @staticmethod
    def write_epoched(subject_id, session_id, epoched_feature, feature_type):
        feature_path = PathService.get_epoched_file_path(subject_id, session_id, feature_type)
        if(feature_type.name == FeatureType.epoched.name):
            epoched_feature.to_csv(feature_path, index=False)
        else:
            np.savetxt(feature_path, epoched_feature, fmt='%f')
    
    @staticmethod
    def write_nightly(nightly_dataframe):
        nightly_feature_path = PathService.get_nightly_file_path()
        nightly_dataframe.to_csv(nightly_feature_path, index=False)