import sys
# caution: path[0] is reserved for script path (or '' in REPL)
# We are inserting the sleepclassifiers as a module 
sys.path.insert(1, '../..')

from source.constants import Constants
from source.preprocessing.activity_count.activity_count_feature_service import ActivityCountFeatureService
from source.preprocessing.heart_rate.heart_rate_feature_service import HeartRateFeatureService
from source.preprocessing.raw_data_processor import RawDataProcessor
from source.preprocessing.time.time_based_feature_service import TimeBasedFeatureService
from source.preprocessing.clustering.clustering_feature_service import ClusteringFeatureService
from source.analysis.setup.feature_type import FeatureType
from source.data_services.data_service import DataService
from source.data_services.data_loader import DataLoader
from source.data_services.data_writer import DataWriter

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


        # Writing all features to their files
        DataWriter.write_epoched(subject_id, session_id, cosine_feature, FeatureType.epoched_cosine)
        DataWriter.write_epoched(subject_id, session_id, time_feature, FeatureType.epoched_time)
        DataWriter.write_epoched(subject_id, session_id, count_feature, FeatureType.epoched_count)
        DataWriter.write_epoched(subject_id, session_id, heart_rate_feature, FeatureType.epoched_heart_rate)
                                     

        
