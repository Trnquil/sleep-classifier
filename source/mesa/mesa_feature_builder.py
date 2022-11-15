from source.mesa.mesa_heart_rate_service import MesaHeartRateService
from source.preprocessing.heart_rate.heart_rate_feature_service import HeartRateFeatureService
from source.mesa.mesa_actigraphy_service import MesaActigraphyService
from source.preprocessing.raw_data_processor import RawDataProcessor

import numpy as np

class MesaFeatureBuilder(object):
    
    @staticmethod
    def build():
        
        subject_id='0001'
        hr_collection = MesaHeartRateService.load(subject_id)
        count_collection = MesaActigraphyService.load(subject_id)
        
        valid_epochs = RawDataProcessor.get_valid_epochs([hr_collection])
        
        hr_features = HeartRateFeatureService.build_from_collection(hr_collection, valid_epochs)
        hr_mean = np.mean(hr_features.iloc[:,1:], axis=0)
        hr_std = np.std(hr_features.iloc[:,1:], axis=0)
        
        count_feature = count_collection
        count_std = np.std(count_feature.iloc[:,1:], axis=0)
        
        hr_features.iloc[:,1:] = (hr_features.iloc[:,1:] - hr_mean)/hr_std  
        # Normalizing count feature, the first row is a timestamp
        count_feature.iloc[:,1:] = count_feature.iloc[:,1:]/count_std
        
        pass
        