from source.analysis.setup.clustering_algorithm import ClusteringAlgorithm
from source.analysis.setup.upsampling_technique import UpsamplingTechnique
from source.data_services.dataset import DataSet
from source.analysis.setup.feature_type import FeatureType
from source.constants import Constants
from pprint import pprint
from source.figures_saver import FiguresSaver


class RunnerParameters(object):
    PROCESS_USI_BVP_SEGMENTWISE = False
    
    #This does not apply to GEMINI, as it has got its own Configuration File
    NUMBER_OF_CLUSTERS = 6
    
    # If we have already trained a GEMINI model, we can set this to false. Models are saved after each training.
    # If no model has been trained so far, training will happen even if GEMINI_TRAIN is set to false
    # The most recently trained model is used for evaluation
    GEMINI_TRAIN = True
    # We are train on every stride-th epoch to conserve computing power
    GEMINI_STRIDE = 4
    
    # Maximum number of epochs we want to train on
    GEMINI_TRAIN_EPOCHS_MAX = 10000
    
    CLUSTERING_DATASETS = [DataSet.mesa, DataSet.usi]
    
    # The following three features need to have compatible types (like epoched_ibi and epoched_ibi_from_ppg)
    CLUSTER_FEATURES_USI = [FeatureType.epoched_ibi_from_ppg]
    CLUSTER_FEATURES_MESA = [FeatureType.epoched_ibi_from_ppg]
    CLUSTER_FEATURES_MSS = [FeatureType.epoched_ibi]
    
    CLUSTERING_PER_SUBJECT_NORMALIZATION = False  # True: normalize clustering features over subjects, 
                                                 # False: normalize clustering features over all data  
    PCA_REDUCTION = False 
    PCA_COMPONENTS = 5
    
    NIGHTLY_CLUSTER_NORMALIZATION = not CLUSTERING_PER_SUBJECT_NORMALIZATION
    USE_NIGHTLY_NORMALIZED = True
    UPSAMPLING_TECHNIQUE = UpsamplingTechnique.SMOTE
    
    NIGHTLY_FEATURES = [FeatureType.nightly_cluster_kmeans,
                        FeatureType.nightly_cluster_gmm,
                        FeatureType.nightly_cluster_GEMINI,
                        FeatureType.nightly_hr, 
                        FeatureType.nightly_normalized_hr,
                        FeatureType.nightly_count,
                        FeatureType.nightly_ibi,
                        FeatureType.nightly_ibi_from_ppg]
    
    ANALYSIS_FEATURES_USI = [[FeatureType.nightly_cluster_kmeans],
                             [FeatureType.nightly_cluster_gmm],
                             [FeatureType.nightly_cluster_GEMINI],
                             [FeatureType.nightly_cluster_kmeans, FeatureType.nightly_cluster_gmm, FeatureType.nightly_cluster_GEMINI],
                            [FeatureType.nightly_cluster_kmeans, FeatureType.nightly_hr],
                            [FeatureType.nightly_cluster_kmeans, FeatureType.nightly_normalized_hr],
                            [FeatureType.nightly_ibi],
                            [FeatureType.nightly_ibi_from_ppg],
                            [FeatureType.nightly_cluster_kmeans, FeatureType.nightly_ibi, FeatureType.nightly_count],
                            [FeatureType.nightly_cluster_kmeans, FeatureType.nightly_normalized_hr, FeatureType.nightly_count],
                            [FeatureType.nightly_cluster_kmeans, 
                                             FeatureType.nightly_hr, 
                                             FeatureType.nightly_normalized_hr,
                                             FeatureType.nightly_count,
                                             FeatureType.nightly_ibi,
                                             FeatureType.nightly_ibi_from_ppg]
                            ]
    
    ANALYSIS_FEATURES_MSS = [[FeatureType.nightly_cluster_kmeans],
                            [FeatureType.nightly_ibi],
                            [FeatureType.nightly_cluster_kmeans, FeatureType.nightly_ibi, FeatureType.nightly_count]]
    
        
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


        
