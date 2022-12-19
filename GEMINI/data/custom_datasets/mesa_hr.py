from source.mesa.mesa_heart_rate_service import MesaHeartRateService
from source.mesa.mesa_actigraphy_service import MesaActigraphyService
from source.mesa.mesa_psg_service import MesaPSGService
from source.preprocessing.interval import Interval
from source.preprocessing.feature_service import FeatureService
from source.preprocessing.epoch import Epoch
from source.preprocessing.collection_service import CollectionService
from source.mesa.mesa_ppg_service import MesaPPGService
from source.preprocessing.bvp_service import BvpService
from source.mesa.mesa_data_service import MesaDataService
from source.runner_parameters import RunnerParameters

import numpy as np
import torch


def get_dataset(**kwargs):
        
        # We are looking at every stride-th epoch to conserve computing power
        stride = RunnerParameters.GEMINI_STRIDE
        
        # Number uf total epochs we want to train with
        train_epochs_count = RunnerParameters.GEMINI_TRAIN_EPOCHS_MAX
        
        subject_ids = MesaDataService.get_all_subject_ids()
        
        X = np.zeros((train_epochs_count, 500))
        
        # The ys will stay 0, they are simply required to interface with GEMINI
        ys = np.zeros(train_epochs_count)
        

        count_total = 0
        for subject_id in subject_ids:
            
            if(count_total >= train_epochs_count):
                break
                                    
            raw_labeled_sleep = MesaPSGService.load_raw(subject_id)
            heart_rate_collection = MesaHeartRateService.load_raw(subject_id)

            interval = Interval(start_time=0, end_time=np.shape(raw_labeled_sleep)[0])
            heart_rate_collection = CollectionService.crop(heart_rate_collection, interval)
            
            valid_epochs = []
            
            count_subject = 0
            for timestamp in range(interval.start_time, interval.end_time, Epoch.DURATION):
                
                if count_subject % stride == 0:
                    
                    if(count_total >= train_epochs_count):
                        break
                                  
                    epoch = Epoch(timestamp=timestamp, index=len(valid_epochs))
                                                                   
                    heart_rate_indices = FeatureService.get_window(heart_rate_collection.timestamps, epoch)
    
                    if 0 not in heart_rate_collection.values[heart_rate_indices]:
                        valid_epochs.append(epoch)
                        hr_values = heart_rate_collection.values[heart_rate_indices].squeeze()
                        if(len(hr_values) >= 500):
                            X[count_total,:] = hr_values[:500]
                            count_total += 1
                            
                count_subject += 1
                      
        # Making sure that we only take filled in data
        X = X[:count_total,:]
        ys = ys[:count_total]
        
        X = np.expand_dims(X, axis=1)
        return torch.utils.data.TensorDataset(torch.Tensor(X),torch.Tensor(ys).long())
    
def get_train_dataset(**kwargs):
    return get_dataset(**kwargs)
              
                    