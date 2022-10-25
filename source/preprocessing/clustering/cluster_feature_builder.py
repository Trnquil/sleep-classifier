from source.analysis.setup.feature_type import FeatureType
from source.data_service import DataService
from source.constants import Constants
from matplotlib import cm

import numpy as np
from matplotlib import pyplot as plt



class ClusterFeatureBuilder(object):

    @staticmethod
    def build(subject_id, session_id, clustering_model):
        
        if Constants.VERBOSE:
            print("Predicting clusters...")
        
        # TODO: I need to implement this in a cleaner way as to avoid making mistakes
        features = np.stack([DataService.load_feature_raw(subject_id, session_id, FeatureType.epoched_count),
                    DataService.load_feature_raw(subject_id, session_id, FeatureType.epoched_heart_rate),
                    # DataService.load_feature_raw(subject_id, session_id, FeatureType.epoched_cosine),
                    # DataService.load_feature_raw(subject_id, session_id, FeatureType.epoched_time)
                    ]).transpose().squeeze()
        
        clusters = clustering_model.predict(features)
        
        
        ### PLOTTING
        
        # timestamps = np.arange(0, len(clusters)*30/3600, 30/3600, dtype=float)
        # plt.scatter(timestamps, clusters, color=cm.cool(30*np.abs(clusters)), edgecolors='none')
        # plt.show()
        # plt.clf()

        # Writing all features to their files
        DataService.write_epoched(subject_id, session_id, clusters, FeatureType.epoched_cluster)
                                     

        
