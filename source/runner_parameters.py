from source.analysis.setup.clustering_algorithm import ClusteringAlgorithm
from source.data_services.dataset import DataSet
from source.analysis.setup.feature_type import FeatureType

class RunnerParameters(object):
    CLUSTERING_ALGO = ClusteringAlgorithm.KMeans
    NUMBER_OF_CLUSTERS = 5
    CLUSTERING_DATASETS = [DataSet.mesa]
    CLUSTERING_FEATURES = [FeatureType.epoched_ibi_from_ppg, FeatureType.epoched_count]
    CLUSTERING_PER_SUBJECT_NORMALIZATION = False  # True: normalize clustering features over subjects, 
                                                  # False: normalize clustering features over all data
    NIGHTLY_CLUSTER_NORMALIZATION = not CLUSTERING_PER_SUBJECT_NORMALIZATION
    USE_NIGHTLY_NORMALIZED = True
    UPSAMPLE_NIGHTLY = True
    NIGHTLY_FEATURES = [FeatureType.nightly_cluster, 
                        FeatureType.nightly_hr, 
                        FeatureType.nightly_normalized_hr,
                        FeatureType.nightly_count,
                        FeatureType.nightly_ibi,
                        FeatureType.nightly_ibi_from_ppg]