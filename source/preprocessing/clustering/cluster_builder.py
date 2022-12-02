from source.analysis.setup.feature_type import FeatureType
from source.data_services.data_service import DataService
from source.constants import Constants
from source.data_services.data_writer import DataWriter
from source.data_services.dataset import DataSet
from source.preprocessing.clustering.cluster_feature_service import ClusterFeatureService
from source.data_services.data_frame_loader import DataFrameLoader
from source.exception_logger import ExceptionLogger

from matplotlib import cm
import numpy as np
from matplotlib import pyplot as plt
import umap
import pandas as pd
import sys



class ClusterBuilder(object):

    @staticmethod
    def build(subject_id, session_id, dataset, clustering_model):
        
        try:
    
            # TODO: I need to implement this in a cleaner way as to avoid making mistakes
            features_df = DataFrameLoader.load_feature_dataframe(subject_id, session_id, [FeatureType.cluster_features], dataset)
            data = features_df.to_numpy().squeeze()
            features = data[:,1:].squeeze()
            timestamps = data[:,0].squeeze()
            
            clusters = clustering_model.predict(features)
            
            
            ### PLOTTING
            
            if Constants.MAKE_PLOTS_PREPROCESSING and dataset.name == DataSet.usi.name:
                plt.scatter((timestamps - timestamps[0])/3600, clusters, color=cm.cool(30*np.abs(clusters)), edgecolors='none')
                plt.savefig(str(Constants.FIGURE_FILE_PATH) + "/clusters/" + subject_id + "_" + session_id)
                plt.clf()
                
                if(features.shape[0] > 20):
                    reducer = umap.UMAP()
                    embedding = reducer.fit_transform(features)
                    plt.scatter(embedding[:, 0], embedding[:, 1], c=clusters, cmap='Spectral', s=8)
                    plt.savefig(str(Constants.FIGURE_FILE_PATH) + "/umap/" + subject_id + "_" + session_id)
                    plt.clf()
                
                
    
            # Writing all features to their files
            timestamped_clusters = np.stack([timestamps, clusters], axis=1)
            clusters_df = pd.DataFrame(timestamped_clusters)
            clusters_df.columns = ["epoch_timestamp", "cluster"]
            DataWriter.write_cluster(clusters_df, subject_id, session_id, FeatureType.cluster, dataset)
        except:
            ExceptionLogger.append_exception(subject_id, session_id, "Nightly", dataset.name, sys.exc_info()[0])
            print("Error: ", sys.exc_info()[0], " while building clusters for subject " + str(subject_id), ", session " + str(session_id))


        
