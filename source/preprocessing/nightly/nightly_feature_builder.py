import sys

from source.preprocessing.clustering.cluster_nightly_feature_service import ClusterNightlyFeatureService
from source.preprocessing.ibi.ibi_nightly_feature_service import IbiNightlyFeatureService
from source.preprocessing.ibi_mss_nightly_feature_service import IbiMssNightlyFeatureService
from source.data_services.data_service import DataService
from source.analysis.setup.feature_type import FeatureType
from source.preprocessing.built_service import BuiltService
from source.data_services.data_writer import DataWriter
from source.preprocessing.activity_count.activity_count_nightly_feature_service import ActivityCountNightlyFeatureService
from source.preprocessing.heart_rate.heart_rate_nightly_feature_service import HeartRateNightlyFeatureService
from source.data_services.dataset import DataSet
from source.data_services.data_frame_loader import DataFrameLoader
from source.runner_parameters import RunnerParameters
from source.exception_logger import ExceptionLogger
from source.analysis.setup.upsampling_technique import UpsamplingTechnique
from source.preprocessing.nightly.upsampler import Upsampler
from source.preprocessing.nightly.feature_space_reducer import FeatureSpaceReducer
from source.analysis.setup.clustering_algorithm import ClusteringAlgorithm
from source.preprocessing.sleep_wake import SleepWake
from source.preprocessing.sleep_session_services.sleep_session_service import SleepSessionService

import pandas as pd
import numpy as np
from tqdm import tqdm


class NightlyFeatureBuilder(object):

    @staticmethod
    def build(dataset):

        subject_index = 0
        
        subject_ids = BuiltService.get_built_subject_ids(FeatureType.epoched, SleepWake.sleep, dataset)
        
        with tqdm(subject_ids, colour='green', unit='subject') as t:
            for subject_id in t:
                t.set_description("Building " + dataset.name +" Nightly Features")
                
                session_index = 0
                for session_id in BuiltService.get_built_sleepsession_ids(subject_id, FeatureType.epoched, SleepWake.sleep, dataset):
                    if dataset.name == DataSet.usi.name:
                        feature_dict = NightlyFeatureBuilder.build_feature_dict_usi(subject_id, session_id)
                    elif dataset.name == DataSet.mss.name:
                        feature_dict = NightlyFeatureBuilder.build_feature_dict_mss(subject_id, session_id)
                    
                    feature_row = pd.DataFrame(feature_dict)
                    
                    if session_index == 0:
                        subject_dataframe = feature_row
                    else:
                        subject_dataframe = pd.concat([subject_dataframe, feature_row], axis=0)
                        
                    session_index += 1
                
                #Normalizing features across the subject and filling 0 for features with std 0
                
                if RunnerParameters.NIGHTLY_CLUSTER_NORMALIZATION:
                    # regex for normalizing all features
                    regex="^((?!subject_|session_|sleep_quality).)*$"
                else:
                    # regex for normalizing all features except gmm and kmeans cluster features
                    regex="^((?!gmm_c_|kmeans_c_|subject_|session_|sleep_quality).)*$"
                    
                    
                subject_mean = np.mean(subject_dataframe.filter(regex=regex), axis=0)
                subject_std = np.std(subject_dataframe.filter(regex=regex), axis=0)*2
                
                subject_dataframe_normalized = subject_dataframe.copy()
                
                # normalized subject dataframe
                subject_dataframe_normalized[subject_dataframe_normalized.filter(regex=regex).columns] = (subject_dataframe_normalized.filter(regex=regex) - subject_mean)/subject_std
                subject_dataframe_normalized = subject_dataframe_normalized.fillna(0)
                
                if subject_index == 0:
                    nightly_dataframe = subject_dataframe
                    nightly_dataframe_normalized = subject_dataframe_normalized
                else:
                    nightly_dataframe = pd.concat([nightly_dataframe, subject_dataframe], axis=0)
                    nightly_dataframe_normalized = pd.concat([nightly_dataframe_normalized, subject_dataframe_normalized], axis=0)
                
                subject_index += 1
        
        # Here we upsample according to the technique set in Runner Parameters
        if RunnerParameters.UPSAMPLING_TECHNIQUE.name == UpsamplingTechnique.random_duplication.name:
            nightly_dataframe = Upsampler.random_duplication_upsampling(nightly_dataframe)
            nightly_dataframe_normalized = Upsampler.random_duplication_upsampling(nightly_dataframe_normalized)
            
        elif RunnerParameters.UPSAMPLING_TECHNIQUE.name == UpsamplingTechnique.SMOTE.name:
            nightly_dataframe =  Upsampler.smote_upsampling(nightly_dataframe)
            nightly_dataframe_normalized = Upsampler.smote_upsampling(nightly_dataframe_normalized)
        
        # Applying PCA Reduction if it is set to True in RunnerParameters
        if RunnerParameters.PCA_REDUCTION:
            nightly_dataframe = FeatureSpaceReducer.PCA(nightly_dataframe)
            nightly_dataframe_normalized = FeatureSpaceReducer.PCA(nightly_dataframe_normalized)

        DataWriter.write_nightly(nightly_dataframe, FeatureType.nightly, dataset)
        DataWriter.write_nightly(nightly_dataframe_normalized, FeatureType.normalized_nightly, dataset)

        
        
    @staticmethod
    def build_feature_dict_usi(subject_id, session_id):
        try:

            cluster_feature_types_sleep = [FeatureType.cluster_gmm, FeatureType.cluster_kmeans, FeatureType.epoched_cluster_GEMINI]
            cluster_feature_types_wake = [FeatureType.cluster_gmm, FeatureType.cluster_kmeans]
                
            clusters_sleep = DataFrameLoader.load_feature_dataframe(subject_id, session_id, cluster_feature_types_sleep, SleepWake.sleep, DataSet.usi)
            clusters_wake = DataFrameLoader.load_feature_dataframe(subject_id, session_id, cluster_feature_types_wake, SleepWake.wake, DataSet.usi)
            cluster_timestamps_sleep = clusters_sleep['epoch_timestamp']
            cluster_timestamps_wake = clusters_wake['epoch_timestamp']
            
            subject_session_dict = {'subject_id': subject_id, 'session_id': session_id}
            
            merged_dict = subject_session_dict
            
            # Building Nightly Wake GMM Cluster Features
            if(FeatureType.nightly_cluster_wake_gmm.name in FeatureType.get_names(RunnerParameters.NIGHTLY_FEATURES)):
                cluster_features_gmm_dict = ClusterNightlyFeatureService.build_feature_dict(subject_id, session_id, FeatureType.cluster_gmm, SleepWake.wake, DataSet.usi)
                cluster_features_gmm_dict = {"wake_gmm_" + str(key): val for key, val in cluster_features_gmm_dict.items()}
                merged_dict = merged_dict | cluster_features_gmm_dict
                
            # Building Nightly Wake KMeans Cluster Features
            if(FeatureType.nightly_cluster_wake_kmeans.name in FeatureType.get_names(RunnerParameters.NIGHTLY_FEATURES)):
                cluster_features_kmeans_dict = ClusterNightlyFeatureService.build_feature_dict(subject_id, session_id, FeatureType.cluster_kmeans, SleepWake.wake, DataSet.usi)
                cluster_features_kmeans_dict = {"wake_kmeans_" + str(key): val for key, val in cluster_features_kmeans_dict.items()}
                merged_dict = merged_dict | cluster_features_kmeans_dict
            
            # Building Nightly Sleep GMM Cluster Features
            if(FeatureType.nightly_cluster_sleep_gmm.name in FeatureType.get_names(RunnerParameters.NIGHTLY_FEATURES)):
                cluster_features_gmm_dict = ClusterNightlyFeatureService.build_feature_dict(subject_id, session_id, FeatureType.cluster_gmm, SleepWake.sleep, DataSet.usi)
                cluster_features_gmm_dict = {"sleep_gmm_" + str(key): val for key, val in cluster_features_gmm_dict.items()}
                merged_dict = merged_dict | cluster_features_gmm_dict
                
            # Building Nightly Sleep KMeans Cluster Features
            if(FeatureType.nightly_cluster_sleep_kmeans.name in FeatureType.get_names(RunnerParameters.NIGHTLY_FEATURES)):
                cluster_features_kmeans_dict = ClusterNightlyFeatureService.build_feature_dict(subject_id, session_id, FeatureType.cluster_kmeans, SleepWake.sleep, DataSet.usi)
                cluster_features_kmeans_dict = {"sleep_kmeans_" + str(key): val for key, val in cluster_features_kmeans_dict.items()}
                merged_dict = merged_dict | cluster_features_kmeans_dict
            
            # Building Nightly Sleep GEMINI Cluster Features for USI DataSet.usi
            if(FeatureType.nightly_cluster_sleep_GEMINI.name in FeatureType.get_names(RunnerParameters.NIGHTLY_FEATURES)):
                cluster_features_GEMINI_dict = ClusterNightlyFeatureService.build_feature_dict(subject_id, session_id, FeatureType.epoched_cluster_GEMINI, SleepWake.sleep, DataSet.usi)
                cluster_features_GEMINI_dict = {"sleep_GEMINI_" + str(key): val for key, val in cluster_features_GEMINI_dict.items()}
                merged_dict = merged_dict | cluster_features_GEMINI_dict
            
            # Building Nightly Sleep GMM Cluster Features
            if(FeatureType.nightly_cluster_sleep_gmm.name in FeatureType.get_names(RunnerParameters.NIGHTLY_FEATURES)):
                cluster_features_gmm_dict = ClusterNightlyFeatureService.build_feature_dict(subject_id, session_id, FeatureType.cluster_gmm, SleepWake.sleep, DataSet.usi)
                cluster_features_gmm_dict = {"selfreported_sleep_gmm_" + str(key): val for key, val in cluster_features_gmm_dict.items()}
                merged_dict = merged_dict | cluster_features_gmm_dict
                
            # Building Nightly Sleep KMeans Cluster Features
            if(FeatureType.nightly_cluster_sleep_kmeans.name in FeatureType.get_names(RunnerParameters.NIGHTLY_FEATURES)):
                cluster_features_kmeans_dict = ClusterNightlyFeatureService.build_feature_dict(subject_id, session_id, FeatureType.cluster_kmeans, SleepWake.sleep, DataSet.usi)
                cluster_features_kmeans_dict = {"selfreported_sleep_kmeans_" + str(key): val for key, val in cluster_features_kmeans_dict.items()}
                merged_dict = merged_dict | cluster_features_kmeans_dict
            
            # Building Nightly Sleep GEMINI Cluster Features for USI DataSet.usi
            if(FeatureType.nightly_cluster_sleep_GEMINI.name in FeatureType.get_names(RunnerParameters.NIGHTLY_FEATURES)):
                cluster_features_GEMINI_dict = ClusterNightlyFeatureService.build_feature_dict(subject_id, session_id, FeatureType.epoched_cluster_GEMINI, SleepWake.sleep, DataSet.usi)
                cluster_features_GEMINI_dict = {"selfreported_sleep_GEMINI_" + str(key): val for key, val in cluster_features_GEMINI_dict.items()}
                merged_dict = merged_dict | cluster_features_GEMINI_dict
            
            # Building Nightly Count Features                          
            if(FeatureType.nightly_count.name in FeatureType.get_names(RunnerParameters.NIGHTLY_FEATURES)):
                count_features_dict = ActivityCountNightlyFeatureService.build_feature_dict_from_epoched(subject_id, session_id, SleepWake.sleep, DataSet.usi, cluster_timestamps_sleep)
                merged_dict = merged_dict | count_features_dict

            # Building Nightly Ibi Features           
            if(FeatureType.nightly_ibi_mss.name in FeatureType.get_names(RunnerParameters.NIGHTLY_FEATURES)):
                ibi_mss_features_dict = IbiMssNightlyFeatureService.build_feature_dict_from_epoched(subject_id, session_id, SleepWake.sleep, DataSet.usi, cluster_timestamps_sleep)
                merged_dict = merged_dict | ibi_mss_features_dict
            
            # Building Nightly HR Features     
            if(FeatureType.nightly_hr.name in FeatureType.get_names(RunnerParameters.NIGHTLY_FEATURES)):
                hr_features_dict = HeartRateNightlyFeatureService.build_feature_dict_from_epoched(subject_id, session_id, SleepWake.sleep, DataSet.usi, cluster_timestamps_sleep)
                merged_dict = merged_dict | hr_features_dict
                    
            
            # Building Nightly Ibi Features           
            if(FeatureType.nightly_ibi.name in FeatureType.get_names(RunnerParameters.NIGHTLY_FEATURES)):
                ibi_features_dict = IbiNightlyFeatureService.build_feature_dict_from_epoched(subject_id, session_id, SleepWake.sleep, DataSet.usi, cluster_timestamps_sleep)
                merged_dict = merged_dict | ibi_features_dict
            
            # Building Nightly Ibi from ppg      
            if(FeatureType.nightly_ibi_from_ppg.name in FeatureType.get_names(RunnerParameters.NIGHTLY_FEATURES)):
                ibi_features_from_ppg_dict = IbiNightlyFeatureService.build_feature_dict_from_epoched_ppg(subject_id, session_id, SleepWake.sleep, DataSet.usi, cluster_timestamps_sleep)
                merged_dict = merged_dict | ibi_features_from_ppg_dict
            
            # Getting the sleep duration
            sleepsession = SleepSessionService.load_sleepsession(subject_id, session_id, SleepWake.sleep, DataSet.usi)
            sleep_duration = sleepsession.end_timestamp - sleepsession.start_timestamp
            sleep_duration_dict = {'sleep_duration': sleep_duration}
            merged_dict = merged_dict | sleep_duration_dict
            
            # Building nightly sleep qualiy
            sleepquality_avg = np.mean(DataService.load_feature_raw(subject_id, FeatureType.sleep_quality, SleepWake.sleep, DataSet.usi))
            sleepquality = DataService.load_feature_raw(subject_id, session_id, FeatureType.sleep_quality, SleepWake.sleep, DataSet.usi)
            sleepquality = 0 if sleepquality < sleepquality_avg else 1
            sleepquality_dict = {'sleep_quality': sleepquality}
            
            merged_dict = merged_dict | sleepquality_dict
                  
            
            return merged_dict
        except:
            ExceptionLogger.append_exception(subject_id, session_id, "Nightly", DataSet.usi.name, sys.exc_info()[0])
            print("Skip subject ", str(subject_id), ", session ", str(session_id), " due to ", sys.exc_info()[0])

    @staticmethod
    def build_feature_dict_mss(subject_id, session_id):
        try:
            cluster_feature_types = [FeatureType.cluster_gmm, FeatureType.cluster_kmeans]
                
            clusters = DataFrameLoader.load_feature_dataframe(subject_id, session_id, cluster_feature_types, SleepWake.sleep, DataSet.mss)
            cluster_timestamps = clusters['epoch_timestamp']
            
            subject_session_dict = {'subject_id': subject_id, 'session_id': session_id}
            
            merged_dict = subject_session_dict
            
            # Building Nightly Sleep GMM Cluster Features
            if(FeatureType.nightly_cluster_sleep_gmm.name in FeatureType.get_names(RunnerParameters.NIGHTLY_FEATURES)):
                cluster_features_gmm_dict = ClusterNightlyFeatureService.build_feature_dict(subject_id, session_id, FeatureType.cluster_gmm, SleepWake.sleep, DataSet.mss)
                cluster_features_gmm_dict = {"sleep_gmm_" + str(key): val for key, val in cluster_features_gmm_dict.items()}
                merged_dict = merged_dict | cluster_features_gmm_dict
                
            # Building Nightly Sleep KMeans Cluster Features
            if(FeatureType.nightly_cluster_sleep_kmeans.name in FeatureType.get_names(RunnerParameters.NIGHTLY_FEATURES)):
                cluster_features_kmeans_dict = ClusterNightlyFeatureService.build_feature_dict(subject_id, session_id, FeatureType.cluster_kmeans, SleepWake.sleep, DataSet.mss)
                cluster_features_kmeans_dict = {"sleep_kmeans_" + str(key): val for key, val in cluster_features_kmeans_dict.items()}
                merged_dict = merged_dict | cluster_features_kmeans_dict
            
            # Building Nightly Count Features                          
            if(FeatureType.nightly_count.name in FeatureType.get_names(RunnerParameters.NIGHTLY_FEATURES)):
                count_features_dict = ActivityCountNightlyFeatureService.build_feature_dict_from_epoched(subject_id, session_id, SleepWake.sleep, DataSet.mss, cluster_timestamps)
                merged_dict = merged_dict | count_features_dict

            # Building Nightly Ibi Features           
            if(FeatureType.nightly_ibi_mss.name in FeatureType.get_names(RunnerParameters.NIGHTLY_FEATURES)):
                ibi_mss_features_dict = IbiMssNightlyFeatureService.build_feature_dict_from_epoched(subject_id, session_id, SleepWake.sleep, DataSet.mss, cluster_timestamps)
                merged_dict = merged_dict | ibi_mss_features_dict
            
            # Building Nightly HR Features     
            if(FeatureType.nightly_hr.name in FeatureType.get_names(RunnerParameters.NIGHTLY_FEATURES)):
                hr_features_dict = HeartRateNightlyFeatureService.build_feature_dict_from_epoched(subject_id, session_id, SleepWake.sleep, DataSet.mss, cluster_timestamps)
                merged_dict = merged_dict | hr_features_dict
                    
            # Getting the sleep duration
            sleepsession = SleepSessionService.load_sleepsession(subject_id, session_id, SleepWake.sleep, DataSet.mss)
            sleep_duration = sleepsession.end_timestamp - sleepsession.start_timestamp
            sleep_duration_dict = {'sleep_duration': sleep_duration}
            merged_dict = merged_dict | sleep_duration_dict
            
            # Building nightly sleep qualiy
            sleepquality_avg = np.mean(DataService.load_feature_raw(subject_id, FeatureType.sleep_quality, SleepWake.sleep, DataSet.mss))
            sleepquality = DataService.load_feature_raw(subject_id, session_id, FeatureType.sleep_quality, SleepWake.sleep, DataSet.mss)
            sleepquality = 0 if sleepquality < sleepquality_avg else 1
            sleepquality_dict = {'sleep_quality': sleepquality}
            
            merged_dict = merged_dict | sleepquality_dict
            
            return merged_dict
        except:
            ExceptionLogger.append_exception(subject_id, session_id, "Nightly", DataSet.mss.name, sys.exc_info()[0])
            print("Skip subject ", str(subject_id), ", session ", str(session_id), " due to ", sys.exc_info()[0])
        

    

        
    

        

        
