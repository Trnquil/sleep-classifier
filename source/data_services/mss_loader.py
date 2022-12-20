from source.preprocessing.path_service import PathService
from source.analysis.setup.feature_type import FeatureType
from source.constants import Constants
from source.preprocessing.activity_count.activity_count_service import ActivityCountService
from source.preprocessing.collection import Collection
from source.preprocessing.sleep_session_services.mss_sleep_session_service import MssSleepSessionService
from source.preprocessing.epoch import Epoch

import os
import pandas as pd
import numpy as np

class MssLoader(object):
    
    HR_QUALITY_MIN = 85
    
    MSS_feature_dict = {
        'start_time': 'epoch_timestamp',
        'LF_abs': 'lf',
        'HF_abs': 'hf',
        'freq_ratio': 'lf_hf_ratio',
        'cvnni': 'cvnni',
        'sdnn': 'sdnn',
        'VLF_abs': 'vlf',
        'mean_nn': 'mean_nni',
        'median_nn': 'median_nni',
        'pnn20': 'pnni_20',
        'nn20': 'nni_20',
        'pnn50': 'pnni_50',
        'nn50': 'nni_50',
        'rmssd': 'rmssd',
        'mean_hr': 'mean_hr',
        'max_hr': 'max_hr',
        'min_hr': 'min_hr',
        'std_hr': 'std_hr'
        }
    
    @staticmethod
    def load_epoched_hrv_features(subject_id, session_id):
        raw_path = PathService.get_raw_file_path_mss(subject_id, FeatureType.raw_hrv)
        
        feature_array = pd.read_csv(str(raw_path))
        feature_array = feature_array[MssLoader.MSS_feature_dict.keys()]
        feature_array.columns = MssLoader.MSS_feature_dict.values()
        
        feature_array['epoch_timestamp'] = (feature_array['epoch_timestamp']/1000 - Constants.TIME_CENTER_MSS).astype(int)
        feature_array = feature_array.sort_values(by=['epoch_timestamp'])
        sleepsession = MssSleepSessionService.load_sleepsession(subject_id, session_id)
        
        # Take only features within the right timestamps
        feature_array = feature_array[feature_array['epoch_timestamp'] < sleepsession.end_timestamp] 
        feature_array = feature_array[feature_array['epoch_timestamp'] > sleepsession.start_timestamp]
        
        # Move all timestamps to the correct epoch timestamp
        feature_array['epoch_timestamp'] = feature_array['epoch_timestamp'] - np.mod(feature_array['epoch_timestamp'], Epoch.DURATION)
        
        return feature_array
        
    @staticmethod
    def load_raw_motion(subject_id):
        
        raw_path = PathService.get_raw_file_path_mss(subject_id, FeatureType.raw_acc)
        feature_array = pd.read_csv(str(raw_path))
        feature_array = feature_array[['timestamp', 'x', 'y', 'z']]
        feature_array['timestamp'] = feature_array['timestamp'] - Constants.TIME_CENTER_MSS
        feature_array = feature_array.sort_values('timestamp', ascending=True)
        
        motion_data = feature_array.to_numpy()
        return Collection(subject_id, motion_data, 0)
    
    @staticmethod
    def load_raw_hr(subject_id):
        
        raw_path = PathService.get_raw_file_path_mss(subject_id, FeatureType.raw_algo1)
        feature_array = pd.read_csv(str(raw_path))
        
        # Throwing away data with low quality
        feature_array = feature_array[(feature_array['hrQuality'] > MssLoader.HR_QUALITY_MIN)]
        feature_array = feature_array[['timestamp', 'hr']]
        feature_array['timestamp'] = feature_array['timestamp'] - Constants.TIME_CENTER_MSS
        feature_array = feature_array.sort_values('timestamp', ascending=True)
        
        ibi_data = feature_array.to_numpy()
        return Collection(subject_id, ibi_data, 0)
        