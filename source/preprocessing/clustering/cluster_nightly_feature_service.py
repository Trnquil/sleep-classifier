from source.data_services.data_loader import DataLoaderfrom source.analysis.setup.feature_type import FeatureTypefrom source.data_services.dataset import DataSetfrom source.data_services.data_frame_loader import DataFrameLoaderimport numpy as npimport pandas as pdclass ClusterNightlyFeatureService(object):        @staticmethod    def build_feature_dict(subject_id, session_id):                clusters = DataFrameLoader.load_feature_dataframe(subject_id, session_id, [FeatureType.cluster], DataSet.usi)        clusters = clusters.to_numpy()                features_dict = ClusterNightlyFeatureService.build_percentage_features(clusters[:,1])        merged_dict = features_dict                                return merged_dict            @staticmethod    def build_percentage_features(cluster_feature):                cluster_feature_len = cluster_feature.shape[0]        c_0 = np.sum(cluster_feature == 0)/cluster_feature_len        c_1 = np.sum(cluster_feature == 1)/cluster_feature_len        c_2 = np.sum(cluster_feature == 2)/cluster_feature_len        c_3 = np.sum(cluster_feature == 3)/cluster_feature_len        c_4 = np.sum(cluster_feature == 4)/cluster_feature_len        c_5 = np.sum(cluster_feature == 5)/cluster_feature_len                features_dictionary = {'c_0': [c_0], 'c_1': [c_1], 'c_2': [c_2], 'c_3': [c_3], 'c_4': [c_4], 'c_5': [c_5]}                return features_dictionary        