from enum import Enum


class FeatureType(Enum):
    raw_hr = "raw heart rate"
    raw_acc = "raw acceleration"
    
    cropped_count = "cropped count"
    cropped_motion = "cropped motion"
    cropped_heart_rate = "cropped heart rate"
    
    epoched_heart_rate = "epoched heart rate"
    epoched_count = "epoched count"
    epoched_cosine = "epoched cosine"
    epoched_motion = "epoched motion"
    epoched_circadian_model = "epoched circadian model"
    epoched_time = "epoched time"
    epoched_cluster = "epoched cluster"
    
    nightly = "nightly"
    
    sleep_quality = "sleep quality"
    
    def get_cropped_names():
        return [FeatureType.cropped_count.name, 
                FeatureType.cropped_motion.name, 
                FeatureType.cropped_heart_rate.name]
    
    def get_epoched_names():
        return [FeatureType.epoched_circadian_model.name, 
                FeatureType.epoched_motion.name, 
                FeatureType.epoched_cluster.name, 
                FeatureType.epoched_cosine.name, 
                FeatureType.epoched_count.name, 
                FeatureType.epoched_heart_rate.name, 
                FeatureType.epoched_time.name]
