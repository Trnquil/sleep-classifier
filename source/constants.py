from source import utils
from pathlib import Path

class Constants(object):
    # WAKE_THRESHOLD = 0.3  # These values were used for scikit-learn 0.20.3, See:
    # REM_THRESHOLD = 0.35  # https://scikit-learn.org/stable/whats_new.html#version-0-21-0
    WAKE_THRESHOLD = 0.5  #
    REM_THRESHOLD = 0.35

    INCLUDE_CIRCADIAN = False
    EPOCH_DURATION_IN_SECONDS = 30
    SECONDS_PER_MINUTE = 60
    SECONDS_PER_DAY = 3600 * 24
    SECONDS_PER_HOUR = 3600
    VERBOSE = True
    MAKE_PLOTS_PREPROCESSING = True
    MESA_DATA_PATH = Path('/local/shared_data/data_mesa')
    CROPPED_FILE_PATH = utils.get_project_root().joinpath('outputs/cropped features/')
    EPOCHED_FILE_PATH = utils.get_project_root().joinpath('outputs/epoched features/')
    CLUSTERS_FILE_PATH = utils.get_project_root().joinpath('outputs/clusters/')
    MESA_FOLDER_NAME = 'mesa'
    MSS_FOLDER_NAME = 'mss'
    USI_FOLDER_NAME = 'usi'
    NIGHTLY_FILE_PATH = utils.get_project_root().joinpath('outputs/nightly features/')
    FIGURE_FILE_PATH = utils.get_project_root().joinpath('figures/')
    LOGS_FILE_PATH = utils.get_project_root().joinpath('logs/')
    LOWER_BOUND = -0.2
    MATLAB_PATH = '/Applications/MATLAB_R2019a.app/bin/matlab'  # Replace with your MATLAB path
    TIME_CENTER_USI = 1614400000.000000
    TIME_CENTER_MSS = 1568600000.000000

    SUBJECT_IDS = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15', 'S16']

