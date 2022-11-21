from source.analysis.setup.feature_type import FeatureType
from source.data_services.data_service import DataService
from source.constants import Constants
from source.data_services.data_writer import DataWriter
from source.data_services.dataset import DataSet


from matplotlib import cm
import numpy as np
from matplotlib import pyplot as plt
import umap



class ClusterFeatureBuilder(object):

    @staticmethod
    def build(subject_id, session_id, dataset, clustering_model):
        
        if Constants.VERBOSE:
            print("Predicting clusters...")
        
        # TODO: I need to implement this in a cleaner way as to avoid making mistakes
        data = DataService.load_feature_raw(subject_id, session_id, FeatureType.epoched, dataset)
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
        DataWriter.write_epoched(clusters, subject_id, session_id, FeatureType.epoched_cluster, dataset)
                                     

        
