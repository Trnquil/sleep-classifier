# caution: path[0] is reserved for script path (or '' in REPL)
# We are inserting the sleepclassifiers as a module 


import os

from source.analysis.setup.feature_type import FeatureType
from source.analysis.setup.subject import Subject
from source.constants import Constants
from source.preprocessing.activity_count.activity_count_feature_service import ActivityCountFeatureService
from source.preprocessing.heart_rate.heart_rate_feature_service import HeartRateFeatureService
from source.preprocessing.time.time_based_feature_service import TimeBasedFeatureService
from source.analysis.setup.sleep_session_service import SleepSessionService
from source.data_services.data_service import DataService
from source.preprocessing.built_service import BuiltService
from source.data_services.dataset import DataSet


class SubjectBuilder(object):
    

    @staticmethod
    def get_subject_dictionary():
        subject_dictionary = {}
        all_subject_ids = BuiltService.get_built_subject_ids(FeatureType.epoched, DataSet.usi)
        for subject_id in all_subject_ids:
            subject_dictionary[subject_id] = SubjectBuilder.build(subject_id)

        return subject_dictionary

    @staticmethod
    def build(subject_id):
        
        feature_dictionary = {}
        
        for feature_type in FeatureType.get_nightly_featuretypes():
            feature = DataService.load_feature_raw(subject_id, feature_type, DataSet.usi)
            feature_dictionary[feature_type.name] = feature
            

        subject = Subject(subject_id=subject_id,
                          feature_dictionary=feature_dictionary)

        # Uncomment to save plots of every subject's data:
        # ax = plt.subplot(5, 1, 1)
        # ax.plot(range(len(feature_hr)), feature_hr)
        # ax = plt.subplot(5, 1, 2)
        # ax.plot(range(len(feature_count)), feature_count)
        # ax = plt.subplot(5, 1, 3)
        # ax.plot(range(len(feature_cosine)), feature_cosine)
        # ax = plt.subplot(5, 1, 4)
        # ax.plot(range(len(feature_circadian)), feature_circadian)
        # ax = plt.subplot(5, 1, 5)
        # ax.plot(range(len(labeled_sleep)), labeled_sleep)
        #
        # plt.savefig(str(Constants.FIGURE_FILE_PATH.joinpath(subject_id + '_applewatch.png')))
        # plt.close()
        return subject
