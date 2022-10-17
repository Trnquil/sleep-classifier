import numpy as np
import pandas as pd
import os

from source.constants import Constants
from source.preprocessing.path_service import PathService


class MotionFeatureService(object):

    @staticmethod
    def load(subject_id, session_id):
        motion_feature_path = MotionFeatureService.get_path(subject_id, session_id)
        feature = pd.read_csv(str(motion_feature_path)).values
        return feature

    @staticmethod
    def get_path(subject_id, session_id):
        return PathService.get_feature_folder_path(subject_id, session_id) + '/motion_feature.out'

    @staticmethod
    def write(subject_id, session_id, feature):
        motion_feature_path = MotionFeatureService.get_path(subject_id, session_id)
        np.savetxt(motion_feature_path, feature, fmt='%f')
