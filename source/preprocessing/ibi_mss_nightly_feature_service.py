from source.data_services.data_service import DataService
from source.analysis.setup.feature_type import FeatureType
from source.data_services.dataset import DataSet
from source.data_services.data_loader import DataLoader

import numpy as np
import pandas as pd

class IbiMssNightlyFeatureService(object):
    
    @staticmethod
    def build_feature_dict_from_epoched(subject_id, session_id, dataset, cluster_timestamps):
        ibi_mss_feature = DataLoader.load_epoched(subject_id, session_id, FeatureType.epoched_ibi_mss, dataset)
        ibi_mss_feature = pd.merge(cluster_timestamps, ibi_mss_feature, how="inner", on=["epoch_timestamp"])
        ibi_mss_feature = ibi_mss_feature.drop(columns=['epoch_timestamp'])
        ibi_mss_feature_avg = pd.DataFrame(np.mean(ibi_mss_feature, axis=0))
        ibi_mss_feature_dict = ibi_mss_feature_avg.to_dict()[0]
        return ibi_mss_feature_dict