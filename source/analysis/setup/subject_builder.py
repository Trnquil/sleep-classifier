# caution: path[0] is reserved for script path (or '' in REPL)
# We are inserting the sleepclassifiers as a module 
import sys
sys.path.insert(1, '/Users/julien/OneDrive/ETH/HS22/Bachelor Thesis/sleep-classifier')


from source.analysis.setup.feature_type import FeatureType
from source.analysis.setup.subject import Subject
from source.constants import Constants
from source.preprocessing.activity_count.activity_count_feature_service import ActivityCountFeatureService
from source.preprocessing.heart_rate.heart_rate_feature_service import HeartRateFeatureService
from source.preprocessing.time.time_based_feature_service import TimeBasedFeatureService


class SubjectBuilder(object):

    #Modified
    @staticmethod
    def get_all_subject_ids():

        subjects_as_strings = ['S01', 'S02', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15', 'S16']
        return subjects_as_strings

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
