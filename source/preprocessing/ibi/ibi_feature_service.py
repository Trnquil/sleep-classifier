import numpy as np
import pandas as pd

from source.preprocessing.feature_service import FeatureService
from source.data_services.dataset import DataSet

from hrvanalysis import *

class IbiFeatureService(object):
    # This controlls what ratio of ibi data there must be inside of a 10-minute window to be accepted 
    DataRatio = 0.95
    # This controls how many datapoints are atleast needed to calculate HRV features
    MinimumValueCount = 256


    @staticmethod
    def build_from_collection(ibi_collection, dataset, valid_epochs):
        ibi_features = []

        i = 0
        for epoch in valid_epochs:
            indices_in_range = FeatureService.get_window(ibi_collection.timestamps, epoch)
            
            ibi_values_in_range = ibi_collection.values[indices_in_range].squeeze()
            ibi_timestamps_in_range = ibi_collection.timestamps[indices_in_range].squeeze()
            
            if ibi_timestamps_in_range.shape[0] == 0:
                continue
            
            ibi_values_delta = np.sum(ibi_values_in_range)
            ibi_timestamps_delta = ibi_timestamps_in_range[-1] - ibi_timestamps_in_range[0]
            
            # We cannot apply this criterion on MSS Data,because MSS IBI is derived from heart rate values
            if(not dataset.name == DataSet.mss.name):
                if(ibi_values_delta/ibi_timestamps_delta < IbiFeatureService.DataRatio):
                    continue
                
            # Applying another criterion
            if(len(ibi_values_in_range) < IbiFeatureService.MinimumValueCount):
                continue
            
            # We need the IBI values in milliseconds, not seconds
            ibi_values = ibi_values_in_range*1000
            

            feature_dict = IbiFeatureService.get_features(ibi_values)
                
            epoch_timestamp_dict = {'epoch_timestamp': epoch.timestamp}
            feature_dict = epoch_timestamp_dict | feature_dict
                     
            # initializing an array filled with nans
            if(i == 0):
                ibi_features = np.full((len(valid_epochs),len(list(feature_dict.keys()))),np.nan)
                
            ibi_features[i,:] = np.array(list(feature_dict.items()))[:,1]
                
            i += 1
        
        if np.any(ibi_features):
            #removing any leftover nans
            ibi_features = ibi_features[~np.isnan(ibi_features).any(axis=1), :]
            columns=np.array(list(feature_dict.items()))[:,0]
            columns = ["ibi_" + column if column != 'epoch_timestamp' else column for column in columns]
            ibi_dataframe = pd.DataFrame(ibi_features, columns=columns)

        else:
            ibi_dataframe = pd.DataFrame([])
        return ibi_dataframe

    @staticmethod
    def get_features(ibi_values):
        feature_dict = get_frequency_domain_features(ibi_values) | get_time_domain_features(ibi_values)
        return feature_dict

