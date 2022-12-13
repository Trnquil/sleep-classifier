from source.analysis.setup.clustering_algorithm import ClusteringAlgorithm
from source.analysis.setup.upsampling_technique import UpsamplingTechnique
from source.data_services.dataset import DataSet
from source.analysis.setup.feature_type import FeatureType
from source.constants import Constants
from pprint import pprint
from source.figures_saver import FiguresSaver


class RunnerParameters(object):
    PROCESS_USI_BVP_SEGMENTWISE = False
    
    CLUSTERING_ALGO = ClusteringAlgorithm.KMeans
    NUMBER_OF_CLUSTERS = 5
    
    CLUSTERING_DATASETS = [DataSet.mesa, DataSet.usi]
    
    # The following three features need to have compatible types (like epoched_ibi and epoched_ibi_from_ppg)
    CLUSTER_FEATURES_USI = [FeatureType.epoched_ibi_from_ppg]
    CLUSTER_FEATURES_MESA = [FeatureType.epoched_ibi_from_ppg]
    CLUSTER_FEATURES_MSS = [FeatureType.epoched_ibi]
    
    CLUSTERING_PER_SUBJECT_NORMALIZATION = True  # True: normalize clustering features over subjects, 
                                                  # False: normalize clustering features over all data
                                                  
    NIGHTLY_CLUSTER_NORMALIZATION = not CLUSTERING_PER_SUBJECT_NORMALIZATION
    USE_NIGHTLY_NORMALIZED = True
    UPSAMPLING_TECHNIQUE = UpsamplingTechnique.SMOTE
    
    NIGHTLY_FEATURES = [FeatureType.nightly_cluster,
                        FeatureType.nightly_hr, 
                        FeatureType.nightly_normalized_hr,
                        FeatureType.nightly_count,
                        FeatureType.nightly_ibi,
                        FeatureType.nightly_ibi_from_ppg]
    
    ANALYSIS_FEATURES_USI = [[FeatureType.nightly_cluster],
                            [FeatureType.nightly_cluster, FeatureType.nightly_hr],
                            [FeatureType.nightly_cluster, FeatureType.nightly_normalized_hr],
                            [FeatureType.nightly_ibi],
                            [FeatureType.nightly_ibi_from_ppg],
                            [FeatureType.nightly_cluster, FeatureType.nightly_ibi, FeatureType.nightly_count],
                            [FeatureType.nightly_cluster, FeatureType.nightly_normalized_hr, FeatureType.nightly_count],
                            [FeatureType.nightly_cluster, 
                                             FeatureType.nightly_hr, 
                                             FeatureType.nightly_normalized_hr,
                                             FeatureType.nightly_count,
                                             FeatureType.nightly_ibi,
                                             FeatureType.nightly_ibi_from_ppg]
                            ]
    
    ANALYSIS_FEATURES_MSS = [[FeatureType.nightly_cluster],
                            [FeatureType.nightly_ibi],
                            [FeatureType.nightly_cluster, FeatureType.nightly_ibi, FeatureType.nightly_count]]
    
        
    def print_settings(dataset):
        path = FiguresSaver.get_figures_path(dataset).joinpath("settings.txt")
            
        with open(path, "w") as log_file:
            for attribute, value in vars(RunnerParameters).items():
                if not callable(value) and not attribute.startswith("__"):
                    log_file.write(str(attribute) + " = ")
                
                    if(type(value) is list):
                        pprint(value, log_file)
                        log_file.write("\n")
                    else:
                        log_file.write(str(value))
                        log_file.write("\n\n")


        
