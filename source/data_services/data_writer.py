from source.preprocessing.path_service import PathService

import numpy as np


class DataWriter(object):
    
    @staticmethod
    def write_cropped(collection, session_id, feature_type):
        output_path = PathService.get_cropped_file_path(collection.subject_id, session_id, feature_type)
        np.savetxt(output_path, collection.data, fmt='%f')
    
    @staticmethod
    def write_epoched(subject_id, session_id, epoched_dataframe, feature_type):
        feature_path = PathService.get_epoched_file_path(subject_id, session_id, feature_type)
        epoched_dataframe.to_csv(feature_path, index=False)
    
    @staticmethod
    def write_nightly(nightly_dataframe):
        nightly_feature_path = PathService.get_nightly_file_path()
        nightly_dataframe.to_csv(nightly_feature_path, index=False)