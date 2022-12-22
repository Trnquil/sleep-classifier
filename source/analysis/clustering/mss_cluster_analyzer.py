import sys
sys.path.insert(1, "../../..")

from source.data_services.data_frame_loader import DataFrameLoader
from source.data_services.dataset import DataSet
from source import utils
from source.analysis.clustering.cluster_analyzer import ClusterAnalyzer
from source.analysis.setup.feature_type import FeatureType
from source.runner_parameters import RunnerParameters
from source.figures_saver import FiguresSaver

import pandas as pd

class MssClusterAnalyzer(object):
    
    @staticmethod
    def filtered_analysis(cluster_feature):
        save_path = FiguresSaver.get_figures_path(DataSet.mss).joinpath("cluster analysis")
        
        cluster_features = RunnerParameters.CLUSTER_FEATURES_MSS.copy()
        cluster_features.append(cluster_feature)
        full_df = DataFrameLoader.load_feature_dataframe(cluster_features, DataSet.mss)
        
        metadata_path = utils.get_project_root().joinpath('data/MS Sleep/metadata.csv')
        metadata_df = pd.read_csv(metadata_path)
        
        female_features, female_clusters = MssClusterAnalyzer.get_features_and_clusters(full_df, metadata_df, 'gender', 'w')
        male_features, male_clusters = MssClusterAnalyzer.get_features_and_clusters(full_df, metadata_df, 'gender', 'm')
        
        features_list = [female_features, male_features]
        clusters_list = [female_clusters, male_clusters]
        
        ClusterAnalyzer.make_mean_comparison_plot(features_list, clusters_list, 'gender', ['w', 'm'], save_path)
        
        patient_features, patient_clusters = MssClusterAnalyzer.get_features_and_clusters(full_df, metadata_df, 'group', 'patient')
        control_features, control_clusters = MssClusterAnalyzer.get_features_and_clusters(full_df, metadata_df, 'group', 'control')
        
        features_list = [patient_features, control_features]
        clusters_list = [patient_clusters, control_clusters]
        
        ClusterAnalyzer.make_mean_comparison_plot(features_list, clusters_list, 'group', ['patient', 'control'], save_path)
        
        age_group_1_features, age_group_1_clusters = MssClusterAnalyzer.get_features_and_clusters_range(full_df, metadata_df, 'age', 0, 18)
        age_group_2_features, age_group_2_clusters = MssClusterAnalyzer.get_features_and_clusters_range(full_df, metadata_df, 'age', 18, 35)
        age_group_3_features, age_group_3_clusters = MssClusterAnalyzer.get_features_and_clusters_range(full_df, metadata_df, 'age', 35, 60)
        age_group_4_features, age_group_4_clusters = MssClusterAnalyzer.get_features_and_clusters_range(full_df, metadata_df, 'age', 60, 99)
        
        features_list = [age_group_1_features, age_group_2_features, age_group_3_features, age_group_4_features]
        clusters_list = [age_group_1_clusters, age_group_2_clusters, age_group_3_clusters, age_group_4_clusters]
        
        ClusterAnalyzer.make_mean_comparison_plot(features_list, clusters_list, 'age', ['0-18', '18-35', '35-60', '60-90'], save_path)
        
    
    @staticmethod
    def get_features_and_clusters(full_df, metadata_df, filter_key, filter_val):
        filtered_df = metadata_df[metadata_df[filter_key].eq(filter_val)]
        filtered_ids = filtered_df['querumID'].dropna().to_numpy().tolist()
        filtered_full_df = full_df[full_df['subject_id'].isin(filtered_ids)]
        filtered_features = filtered_full_df.drop(columns=['epoch_timestamp', 'subject_id', 'session_id', 'cluster'])
        filtered_clusters = filtered_full_df['cluster']
        return filtered_features, filtered_clusters
    
    @staticmethod
    def get_features_and_clusters_range(full_df, metadata_df, filter_key, range_start, range_end):
        filtered_df = metadata_df[metadata_df[filter_key].ge(range_start)]
        filtered_df = filtered_df[filtered_df[filter_key].le(range_end)]
        filtered_ids = filtered_df['querumID'].dropna().to_numpy().tolist()
        filtered_full_df = full_df[full_df['subject_id'].isin(filtered_ids)]
        filtered_features = filtered_full_df.drop(columns=['epoch_timestamp', 'subject_id', 'session_id', 'cluster'])
        filtered_clusters = filtered_full_df['cluster']
        return filtered_features, filtered_clusters