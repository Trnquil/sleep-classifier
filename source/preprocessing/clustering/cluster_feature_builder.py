from source.analysis.setup.feature_type import FeatureType
from source.data_services.data_service import DataService
from source.constants import Constants
from matplotlib import cm

import numpy as np
from matplotlib import pyplot as plt
from source.data_services.data_writer import DataWriter



class ClusterFeatureBuilder(object):

    @staticmethod
    def build(subject_id, session_id, clustering_model):
        
        if Constants.VERBOSE:
            print("Predicting clusters...")
        
        # TODO: I need to implement this in a cleaner way as to avoid making mistakes
        features = DataService.load_feature_raw(subject_id, session_id, FeatureType.epoched)[:,1:].squeeze()
        
        clusters = clustering_model.predict(features)
        
        
        ### PLOTTING
        
        if Constants.MAKE_PLOTS_PREPROCESSING:
            timestamps = np.arange(0, len(clusters)*30/3600, 30/3600, dtype=float)[0:len(clusters)]
            plt.scatter(timestamps, clusters, color=cm.cool(30*np.abs(clusters)), edgecolors='none')
            plt.savefig(str(Constants.FIGURE_FILE_PATH) + "/clusters/" + subject_id + "_" + session_id)
            plt.clf()
            
            

        # Writing all features to their files
        DataWriter.write_epoched(subject_id, session_id, clusters, FeatureType.epoched_cluster)
                                     

        
