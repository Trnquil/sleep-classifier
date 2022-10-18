import sys
# caution: path[0] is reserved for script path (or '' in REPL)
# We are inserting the sleepclassifiers as a module 
sys.path.insert(1, '../..')

from source.constants import Constants
from source.preprocessing.activity_count.activity_count_feature_service import ActivityCountFeatureService
from source.preprocessing.heart_rate.heart_rate_feature_service import HeartRateFeatureService
from source.preprocessing.raw_data_processor import RawDataProcessor
from source.preprocessing.time.time_based_feature_service import TimeBasedFeatureService
from source.preprocessing.clustering_service import ClusteringService

import numpy as np



class FeatureBuilder(object):

    @staticmethod
    def build(subject_id, session_id):
        if Constants.VERBOSE:
            print("Getting valid epochs...")
        valid_epochs = RawDataProcessor.get_valid_epochs(subject_id, session_id)

        if Constants.VERBOSE:
            print("Building features...")
            
        count_feature = ActivityCountFeatureService.build(subject_id, session_id, valid_epochs)
        heart_rate_feature = HeartRateFeatureService.build(subject_id, session_id, valid_epochs)
        
        if Constants.INCLUDE_CIRCADIAN:
            circadian_feature = TimeBasedFeatureService.build_circadian_model(subject_id, session_id, valid_epochs)
            TimeBasedFeatureService.write_circadian_model(subject_id, session_id, circadian_feature)

        cosine_feature = TimeBasedFeatureService.build_cosine(valid_epochs)
        time_feature = TimeBasedFeatureService.build_time(valid_epochs)
        
        # TODO: I need to implement this in a cleaner way as to avoid making mistakes
        features = np.stack((np.array(count_feature).flatten(), 
                                   np.array(heart_rate_feature).flatten(), 
                                   cosine_feature, 
                                   time_feature)).transpose()
        
        clusters = ClusteringService.get_predictions(features)


        # Writing all features to their files
        TimeBasedFeatureService.write_cosine(subject_id, session_id, cosine_feature)
        TimeBasedFeatureService.write_time(subject_id, session_id, time_feature)
        ActivityCountFeatureService.write(subject_id, session_id, count_feature)
        HeartRateFeatureService.write(subject_id, session_id, heart_rate_feature)
        ClusteringService.write(subject_id, session_id, clusters)
                                     

        
