import sys
# caution: path[0] is reserved for script path (or '' in REPL)
# We are inserting the sleepclassifier as a module to be able to access source.<xxx>
sys.path.insert(1, '../..')

import time

from source.analysis.setup.subject_builder import SubjectBuilder
from source.constants import Constants
from source.preprocessing.activity_count.activity_count_service import ActivityCountService
from source.preprocessing.feature_builder import FeatureBuilder
from source.preprocessing.nightly_feature_builder import NightlyFeatureBuilder
from source.preprocessing.raw_data_processor import RawDataProcessor
from source.preprocessing.time.circadian_service import CircadianService


def run_preprocessing():
    start_time = time.time()

    subject_set = SubjectBuilder.get_all_subject_ids()
    for subject in subject_set:
        print("Cropping data from subject " + str(subject) + "...")
        RawDataProcessor.crop_all(str(subject))

    if Constants.INCLUDE_CIRCADIAN:
        ActivityCountService.build_activity_counts()  # This uses MATLAB, but has been replaced with a python implementation
        CircadianService.build_circadian_model()      # Both of the circadian lines require MATLAB to run
        CircadianService.build_circadian_mesa()       # INCLUDE_CIRCADIAN = False by default because most people don't have MATLAB

        
    # Only building features for subjects and sleepsession for which folders exist
    subject_sleepsession_dictionary = SubjectBuilder.get_built_subject_and_sleepsession_ids()
    for subject in subject_sleepsession_dictionary.keys():
        for session in subject_sleepsession_dictionary[subject]:
            FeatureBuilder.build(subject, session)
            NightlyFeatureBuilder.build(subject, session)
            
    

    end_time = time.time()
    print("Execution took " + str((end_time - start_time) / 60) + " minutes")


run_preprocessing()

# for subject_id in subject_ids:
    # DataPlotBuilder.make_data_demo(subject_id, False)
