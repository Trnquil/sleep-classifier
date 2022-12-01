from source.analysis.setup.clustering_algorithm import ClusteringAlgorithm
from source.data_services.dataset import DataSet
from source.analysis.setup.feature_type import FeatureType

class RunnerParameters(object):
    CLUSTERING_ALGO = ClusteringAlgorithm.KMeans
    NUMBER_OF_CLUSTERS = 6
    CLUSTERING_DATASETS = [DataSet.mesa]
    CLUSTERING_FEATURES = [FeatureType.epoched_hr, FeatureType.epoched_count]
    USE_NIGHTLY_NORMALIZED = True
    UPSAMPLE_NIGHTLY = False
    