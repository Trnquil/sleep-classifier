from matplotlib import pyplot as plt

from source.data_services.dataset import DataSet
from source.data_services.data_loader import DataLoader
from source.analysis.setup.feature_type import FeatureType
from source.preprocessing.clustering.cluster_feature_service import ClusterFeatureService
from source.constants import Constants
from source.figures_saver import FiguresSaver

from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix

import pandas as pd
import numpy as np
import umap


class ClusterAnalyzer(object):
    
    @staticmethod
    def analyze(features, clusters, labels):

        ClusterAnalyzer.make_confusion_matrix(labels, clusters, str(FiguresSaver.get_figures_path(DataSet.usi)) + "/cluster analysis/confusion_matrix")
        ClusterAnalyzer.make_mean_plot(features, clusters, str(FiguresSaver.get_figures_path(DataSet.usi)) + "/cluster analysis/means")
        ClusterAnalyzer.make_umap(features, clusters, FiguresSaver.get_figures_path(DataSet.usi) + "/cluster analysis/umap")

    @staticmethod
    def make_confusion_matrix(labels, clusters, save_path):
        # Making a confusion matrix between predicted classes and class labels
        confusion_matrix = metrics.confusion_matrix(labels.to_numpy(), clusters.to_numpy())
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix)
        cm_display.plot()
        plt.savefig(save_path)
        plt.clf()

    @staticmethod
    def make_mean_plot(features, clusters, save_path):
        cluster_means_df = ClusterAnalyzer.make_cluster_means_df(features, clusters)
        feature_count = features.shape[1]
        cluster_means_df.plot(kind='bar', subplots=True, figsize = (5,3*feature_count))
        plt.savefig(save_path)
        plt.clf()
        
    @staticmethod
    def make_mean_comparison_plot(features_list, clusters_list, group_name, group_labels, save_path):
        groups_count = len(clusters_list)
        feature_count = features_list[0].shape[1]
        
        cluster_means_list = []
        for i in range(groups_count):
            features_df = features_list[i]
            clusters_df = clusters_list[i]
            cluster_means_df = ClusterAnalyzer.make_cluster_means_df(features_df, clusters_df)
            cluster_means_list.append(cluster_means_df)
        
        for i in range(feature_count):
            for j in range(groups_count):
                if j == 0:
                    feature_means_df = cluster_means_list[j].iloc[:,i]
                else:
                    feature_means_df = pd.concat([feature_means_df, cluster_means_list[j].iloc[:,i]], axis=1)
                    
            feature_means_df.columns = group_labels
            feature_means_df.plot(kind='bar', subplots=False, figsize = (5,3))
            title = cluster_means_list[0].columns[i]
            plt.title(title)
            plt.savefig(save_path.joinpath(group_name + "_" + title))
            plt.clf()
        
    @staticmethod
    def make_cluster_means_df(features, clusters):
        full_df = pd.concat([features, clusters], axis=1)
        cluster_means_dict = {}
        unique_clusters = np.unique(clusters.to_numpy())
        for cluster in unique_clusters:
            specific_df = features[full_df['cluster'].eq(cluster)]
            mean = np.mean(specific_df, axis = 0).to_dict()
            cluster_means_dict[cluster] = mean
        
        cluster_means_df = pd.DataFrame(cluster_means_dict).transpose()
        return cluster_means_df
    
    @staticmethod
    def make_umap(features, clusters, save_path):
        reducer = umap.UMAP()
        embedding = reducer.fit_transform(features)
        plt.scatter(embedding[:, 0], embedding[:, 1], c=clusters.values, cmap='Spectral', s=8)
        plt.plot()
        plt.savefig(save_path)
        plt.clf()
    
        
