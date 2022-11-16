import sys
# caution: path[0] is reserved for script path (or '' in REPL)
# We are inserting the sleepclassifier as a module to be able to access source.<xxx>
sys.path.insert(1, '../..')

import time

from source.preprocessing.built_service import BuiltService
from source.constants import Constants
from source.preprocessing.activity_count.activity_count_service import ActivityCountService
from source.preprocessing.epoched_feature_builder import EpochedFeatureBuilder
from source.preprocessing.nightly.nightly_feature_builder import NightlyFeatureBuilder
from source.preprocessing.raw_data_processor import RawDataProcessor
from source.preprocessing.time.circadian_service import CircadianService
from source.preprocessing.clustering.clustering_feature_service import ClusteringFeatureService
from source.preprocessing.clustering.cluster_feature_builder import ClusterFeatureBuilder
from source.analysis.setup.feature_type import FeatureType
from source.data_services.dataset import DataSet


def run_preprocessing():
    start_time = time.time()
    
    build_cropped()
    build_epoched()
    build_clusters()
    build_nightly()            

    end_time = time.time()
    print("Execution took " + str((end_time - start_time) / 60) + " minutes")

def build_cropped():
    subject_set = Constants.SUBJECT_IDS
    for subject in subject_set:
        print("Cropping data from subject " + str(subject) + "...")
        RawDataProcessor.crop_all(str(subject))

    if Constants.INCLUDE_CIRCADIAN:
        ActivityCountService.build_activity_counts()  # This uses MATLAB, but has been replaced with a python implementation
        CircadianService.build_circadian_model()      # Both of the circadian lines require MATLAB to run
        CircadianService.build_circadian_mesa()       # INCLUDE_CIRCADIAN = False by default because most people don't have MATLAB

def build_epoched():
    # Only building features for subjects and sleepsession for which folders exist
    subject_ids = BuiltService.get_built_subject_ids(FeatureType.cropped, DataSet.usi)
    for subject_id in subject_ids:
        EpochedFeatureBuilder.build(subject_id)
            

def build_clusters():
    clustering_model = ClusteringFeatureService.get_fitted_model()
    # Only building features for subjects and sleepsession for which folders exist
    subject_sleepsession_dictionary = BuiltService.get_built_subject_and_sleepsession_ids(FeatureType.epoched, DataSet.usi)
    for subject in subject_sleepsession_dictionary.keys():
        for session in subject_sleepsession_dictionary[subject]:
            ClusterFeatureBuilder.build(subject, session, clustering_model)

def build_nightly():
    NightlyFeatureBuilder.build()
    
    
run_preprocessing()

# for subject_id in subject_ids:
    # DataPlotBuilder.make_data_demo(subject_id, False)
