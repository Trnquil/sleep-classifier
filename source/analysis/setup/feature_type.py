from enum import Enum
import re


class FeatureType(Enum):
    raw_hr = "raw heart rate"
    raw_acc = "raw acceleration"
    raw_ibi = "raw inter beat interval"
    raw_bvp = "raw BVP signal"
    raw_hrv = "raw HRV metrics"
    raw_algo1 = "raw algo 1"
    raw_algo2 = "raw algo 2"
    
    cropped = "cropped"
    
    cropped_count = "cropped count"
    cropped_motion = "cropped motion"
    cropped_ibi_from_ppg = "cropped inter beat intervals from ppg signal"
    cropped_ibi = "cropped inter beat intervals"
    cropped_hr = "cropped heart rate"
    cropped_normalized_hr = "cropped normalized hr"
    
    epoched = "epoched"
    
    epoched_cluster = "epoched cluster"
    epoched_hr = "epoched heart rate"
    epoched_normalized_hr = "epoched normalized hr"
    epoched_count = "epoched count"
    epoched_ibi = "epoched ibi features"
    epoched_ibi_from_ppg = "epoched ibi features from ppg signal"
    epoched_sleep_label = "epoched sleep label"
    
    cluster = "cluster"
    cluster_features = "features on which clustering is performed"
    
    nightly = "nightly"
    normalized_nightly = "normalized nightly"
    
    nightly_cluster = "nightly cluster features"
    nightly_ibi = "nightly ibi features"
    nightly_ibi_from_ppg = "nighty ibi features from ppg"
    nightly_count = "nightly count feature"
    nightly_sleep_quality = "nightly sleep quality label"
    nightly_hr = "nightly heart rate feature"
    nightly_normalized_hr = "nightly normalized heart rate feature"
    
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
    
    def get_names(feature_types):
        names = [feature_type.name for feature_type in feature_types]
        return names