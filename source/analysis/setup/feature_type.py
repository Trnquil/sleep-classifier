from enum import Enum
import re


class FeatureType(Enum):
    raw_hr = "raw heart rate"
    raw_acc = "raw acceleration"
    raw_ibi = "raw inter beat interval"
    
    cropped_count = "cropped count"
    cropped_motion = "cropped motion"
    cropped_heart_rate = "cropped heart rate"
    cropped_ibi = "cropped inter beat intervals"
    normalized_heart_rate = "normalized cropped heart rate"
    
    epoched_heart_rate = "epoched heart rate"
    epoched_count = "epoched count"
    epoched_cosine = "epoched cosine"
    epoched_motion = "epoched motion"
    epoched_circadian_model = "epoched circadian model"
    epoched_time = "epoched time"
    epoched_cluster = "epoched cluster"
    epoched_dataframe = "epoched dataframe"
    
    nightly = "nightly"
    
    nightly_cluster = "nightly cluster features"
    nightly_hr = "nightly heart rate features"
    nightly_sleep_quality = "nightly sleep quality label"
    
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
    
    def get_nightly_featuretypes():
        nightly_features = []
        for feature_type in FeatureType:
            if feature_type.name in FeatureType.get_nightly_names():
                nightly_features.append(feature_type)
        return nightly_features
