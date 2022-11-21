import sys
sys.path.insert(1, '../../..')

from matplotlib import pyplot as plt

from source.data_services.dataset import DataSet
from source.data_services.data_loader import DataLoader
from source.analysis.setup.feature_type import FeatureType
from source.preprocessing.clustering.clustering_feature_service import ClusteringFeatureService
from source.constants import Constants
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix

import pandas as pd
import numpy as np
import umap

class ClusterAnalyzer(object):
    
    @staticmethod
    def analyze(features, clusters, labels):
        
        ClusterAnalyzer.make_confusion_matrix(labels, clusters)
        ClusterAnalyzer.make_mean_plot(features, clusters)
        ClusterAnalyzer.make_umap(features, clusters)

    @staticmethod
    def make_confusion_matrix(labels, clusters):
        # Making a confusion matrix between predicted classes and class labels
        confusion_matrix = metrics.confusion_matrix(labels.to_numpy(), clusters.to_numpy())
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix)
        cm_display.plot()
        plt.savefig(str(Constants.FIGURE_FILE_PATH) + "/cluster analysis/confusion_matrix")
        plt.show()
    
    @staticmethod
    def make_mean_plot(features, clusters):
        full_df = pd.concat([features, clusters], axis=1)
        cluster_means_dict = {}
        unique_clusters = np.unique(clusters.to_numpy())
        for cluster in unique_clusters:
            specific_df = features[full_df['cluster'].eq(cluster)]
            mean = np.mean(specific_df, axis = 0).to_dict()
            cluster_means_dict[cluster] = mean
        
        cluster_means_df = pd.DataFrame(cluster_means_dict).transpose()
        feature_count = features.shape[1]
        cluster_means_df.plot(kind='bar', subplots=True, figsize = (5,3*feature_count))
        plt.savefig(str(Constants.FIGURE_FILE_PATH) + "/cluster analysis/means")
        plt.show()
    
    @staticmethod
    def make_umap(features, clusters):
        reducer = umap.UMAP()
        embedding = reducer.fit_transform(features.iloc[:,1:])
        plt.scatter(embedding[:, 0], embedding[:, 1], c=clusters.values, cmap='Spectral', s=8)
        plt.plot()
        plt.savefig(str(Constants.FIGURE_FILE_PATH) + "/cluster analysis/umap")
        plt.show()
    
        
