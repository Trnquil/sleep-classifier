from source.analysis.setup.clustering_algorithm import ClusteringAlgorithm
from source.data_services.dataset import DataSet
from source.analysis.setup.feature_type import FeatureType
from enum import Enum
from source.constants import Constants

class RunnerParameters(Enum):
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
    
    
    def print_settings():
        with open(Constants.FIGURE_FILE_PATH.joinpath("settings.txt"), "w") as log_file:
            for var in RunnerParameters:
                log_file.write(str(var.name) + " = ")
                
                if(type(var.value) is list):
                    array = var.value
                    if len(array) == 1:
                        log_file.write("[" + str(array[0]) + "]")
                    else:
                        for i in range(len(array)):
                            if i == 0:
                                log_file.write("[\n\t" + str(array[i]) + ", ")
                            elif i == len(array) - 1:
                                log_file.write("\n\t" + str(array[i]) + "\n\t]")
                            else:
                                log_file.write("\n\t" + str(array[i]) + ", ")
                            
                else:
                    log_file.write(str(var.value))
                log_file.write("\n\n")
            
