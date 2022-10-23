# caution: path[0] is reserved for script path (or '' in REPL)
# We are inserting the sleepclassifiers as a module 
import sys
sys.path.insert(1, '/Users/julien/OneDrive/ETH/HS22/Bachelor Thesis/sleep-classifier')

import os

from source.analysis.setup.feature_type import FeatureType
from source.analysis.setup.subject import Subject
from source.constants import Constants
from source.preprocessing.activity_count.activity_count_feature_service import ActivityCountFeatureService
from source.preprocessing.heart_rate.heart_rate_feature_service import HeartRateFeatureService
from source.preprocessing.time.time_based_feature_service import TimeBasedFeatureService
from source.analysis.setup.sleep_session_service import SleepSessionService


class SubjectBuilder(object):

    #Modified
    @staticmethod
    def get_all_subject_ids():

        subjects_as_strings = ['S01', 'S02', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15', 'S16']
        return subjects_as_strings
    
    # returns only subject and sleepsession ids for which there is data
    @staticmethod
    def get_built_subject_and_sleepsession_ids():
        subject_to_session_dictionary = {}
        
        for subject_id in os.listdir(Constants.CROPPED_FILE_PATH):
                if subject_id in SubjectBuilder.get_all_subject_ids():
                    for session_id in os.listdir(Constants.CROPPED_FILE_PATH.joinpath(subject_id)):
                        if session_id in SleepSessionService.get_all_session_ids(subject_id):
                            if subject_id not in subject_to_session_dictionary.keys():
                                subject_to_session_dictionary[subject_id] = []
                                
                            subject_to_session_dictionary[subject_id].append(session_id)
        return subject_to_session_dictionary  

    # returns only sleepsession ids for which there is data
    @staticmethod 
    def get_built_sleepsession_ids(subject_id):
        return SubjectBuilder.get_built_subject_and_sleepsession_ids()[subject_id]    

    # returns only subject ids for which there is data
    @staticmethod 
    def get_built_subject_ids():
        return list(SubjectBuilder.get_built_subject_and_sleepsession_ids().keys())
    
    # returns only subject ids for which there is data
    @staticmethod 
    def get_built_subject_and_sleepsession_count():
        count = 0
        for subject_id in SubjectBuilder.get_built_subject_ids():
            for session_id in SubjectBuilder.get_built_sleepsession_ids(subject_id):
                count += 1
        return count

    @staticmethod
    def get_subject_dictionary():
        subject_dictionary = {}
        all_subject_ids = SubjectBuilder.get_all_subject_ids()
        for subject_id in all_subject_ids:
            subject_dictionary[subject_id] = SubjectBuilder.build(subject_id)

        return subject_dictionary

    @staticmethod
    def build(subject_id):
        feature_count = ActivityCountFeatureService.load(subject_id)
        feature_hr = HeartRateFeatureService.load(subject_id)
        feature_time = TimeBasedFeatureService.load_time(subject_id)
        if Constants.INCLUDE_CIRCADIAN:
            feature_circadian = TimeBasedFeatureService.load_circadian_model(subject_id)
        else:
            feature_circadian = None
        feature_cosine = TimeBasedFeatureService.load_cosine(subject_id)

        feature_dictionary = {FeatureType.count: feature_count,
                              FeatureType.heart_rate: feature_hr,
                              FeatureType.time: feature_time,
                              FeatureType.circadian_model: feature_circadian,
                              FeatureType.cosine: feature_cosine}

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
