import sys

from source.analysis.setup.feature_type import FeatureType
from source.data_services.data_frame_loader import DataFrameLoader
from source.data_services.dataset import DataSet
from source.preprocessing.built_service import BuiltService
from source.data_services.data_writer import DataWriter
from source.preprocessing.path_service import PathService
from source.constants import Constants
from source.runner_parameters import RunnerParameters
from source.exception_logger import ExceptionLogger

import numpy as np
from tqdm import tqdm

class ClusterFeatureBuilder(object):
    
    @staticmethod
    def build(dataset, sleep_wake):
        
        if(dataset.name == DataSet.usi.name):
            cluster_features = RunnerParameters.CLUSTER_FEATURES_USI
        elif(dataset.name == DataSet.mesa.name):
            cluster_features = RunnerParameters.CLUSTER_FEATURES_MESA
        elif(dataset.name == DataSet.mss.name):
            cluster_features = RunnerParameters.CLUSTER_FEATURES_MSS
        
        if not RunnerParameters.CLUSTERING_PER_SUBJECT_NORMALIZATION:
            overall_df = DataFrameLoader.load_feature_dataframe(cluster_features, sleep_wake, dataset)
            overall_mean = np.mean(overall_df.iloc[:,3:], axis=0)
            overall_std = np.std(overall_df.iloc[:,3:], axis=0)
        
        subject_sleepsession_dictionary = BuiltService.get_built_subject_and_sleepsession_ids(FeatureType.epoched, sleep_wake, dataset)
        
        with tqdm(subject_sleepsession_dictionary.keys(), unit='subject', colour='green') as t:
            for subject_id in t:
                try:
                    t.set_description("Building " + dataset.name.upper() + " Cluster Features")
                    
                    subject_df = DataFrameLoader.load_feature_dataframe(subject_id, cluster_features, sleep_wake, dataset)
                    subject_mean = np.mean(subject_df.iloc[:,3:], axis=0)
                    subject_std = np.std(subject_df.iloc[:,3:], axis=0)
                    
                    for session_id in subject_sleepsession_dictionary[subject_id]:
                        try:
                            session_df = DataFrameLoader.load_feature_dataframe(subject_id, session_id, cluster_features, sleep_wake, dataset)
                            normalized_session_df = session_df
                            
                            if RunnerParameters.CLUSTERING_PER_SUBJECT_NORMALIZATION:
                                normalized_session_df.iloc[:,3:] = (normalized_session_df.iloc[:,3:] - subject_mean)/subject_std
                                if(FeatureType.epoched_count in cluster_features):
                                    normalized_session_df[['count']] = normalized_session_df[['count']] + (subject_mean['count']/subject_std['count'])
                            else:
                                normalized_session_df.iloc[:,3:] = (normalized_session_df.iloc[:,3:] - overall_mean)/overall_std
                                if(FeatureType.epoched_count in cluster_features):
                                    normalized_session_df[['count']] = normalized_session_df[['count']] + (overall_mean['count']/overall_std['count'])
                           
                            DataWriter.write_cluster(normalized_session_df, subject_id, session_id, FeatureType.cluster_features, sleep_wake, dataset)
                        except:
                            ExceptionLogger.append_exception(subject_id, session_id, "Cluster Features", dataset.name, sys.exc_info()[0])
                            print("Skip subject ", str(subject_id), ", session ", str(session_id), " due to ", sys.exc_info()[0])
                except:
                    ExceptionLogger.append_exception(subject_id, "N/A", "Cluster Features", dataset.name, sys.exc_info()[0])
                    print("Skip subject ", str(subject_id), " due to ", sys.exc_info()[0])
                    

    