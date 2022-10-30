from enum import Enum
import re


class FeatureType(Enum):
    raw_hr = "raw heart rate"
    raw_acc = "raw acceleration"
    
    cropped_count = "cropped count"
    cropped_motion = "cropped motion"
    cropped_heart_rate = "cropped heart rate"
    normalized_heart_rate = "normalized cropped heart rate"
    
    epoched_heart_rate = "epoched heart rate"
    epoched_count = "epoched count"
    epoched_cosine = "epoched cosine"
    epoched_motion = "epoched motion"
    epoched_circadian_model = "epoched circadian model"
    epoched_time = "epoched time"
    epoched_cluster = "epoched cluster"
    
    nightly = "nightly"
    
    nightly_cluster = "nightly cluster features"
    nightly_hr = "nightly heart rate features"
    
    sleep_quality = "sleep quality"
    
    def get_cropped_names():
        r = re.compile("cropped_.*|normalized_.*")
        feature_type_list = [feature_type.name for feature_type in FeatureType]
        return list(filter(r.match, feature_type_list))
    
    def get_epoched_names():
        r = re.compile("epoched_.*")
        feature_type_list = [feature_type.name for feature_type in FeatureType]
        return list(filter(r.match, feature_type_list))
    
    def get_nightly_names():
        r = re.compile("nightly_.*")
        feature_type_list = [feature_type.name for feature_type in FeatureType]
        return list(filter(r.match, feature_type_list))

