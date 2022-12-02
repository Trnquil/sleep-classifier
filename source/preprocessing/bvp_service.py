import numpy as np

from source.preprocessing.collection import Collection

import heartpy as hp
import scipy.signal as s
import warnings


class BvpService(object):
    IBI_LOWER_BOUND = 0.4
    IBI_UPPER_BOUND = 2
    PROCESSING_WINDOW_DATAPOINTS = 30000
    ACCEPTABLE_DRIFT = 10
    
    @staticmethod
    def get_ibi_from_bvp(bvp_collection):
        
        bvp_values = bvp_collection.values.squeeze()
        bvp_timestamps = bvp_collection.timestamps.squeeze()
        
        # Working on BVP values to produce IBI sequence
        processing_window = BvpService.PROCESSING_WINDOW_DATAPOINTS
        current_height = 0
        ibi_data = []
        
        while current_height < bvp_values.shape[0]:
            
            if(current_height + processing_window >= bvp_values.shape[0]):
                processing_window = bvp_values.shape[0] - current_height
                
            bvp_values_cropped = bvp_values[current_height:(current_height + processing_window)]
            bvp_timestamps_cropped  = bvp_timestamps[current_height:(current_height + processing_window)]
            bvp_data_cropped = np.stack((bvp_timestamps_cropped, bvp_values_cropped), axis=1)
            bvp_collection_cropped = Collection(bvp_collection.subject_id, bvp_data_cropped, bvp_collection.data_frequency)
            
            try:
                ibi_collection_cropped = BvpService.get_ibi_from_bvp_segment(bvp_collection_cropped)
                ibi_data_cropped = ibi_collection_cropped.data
                
                if not np.any(ibi_data):
                    ibi_data = ibi_data_cropped
                else:
                    ibi_data = np.concatenate([ibi_data, ibi_data_cropped], axis = 0)
            except:
                pass

                
            current_height += processing_window
        
        if not BvpService.in_order(ibi_data[:,0]):
            raise Exception("ibi timestamps are not in order, sleep session won't be included")
            
        ibi_collection = Collection(bvp_collection.subject_id, ibi_data, 0)
        return ibi_collection
    
    @staticmethod
    def get_ibi_from_bvp_segment(bvp_collection):
        # Working on BVP values to produce IBI sequence
        bvp_values = bvp_collection.values.squeeze()
        filtered = BvpService.bvp_filter(bvp_values)
        working_data, measures = hp.process(filtered, bvp_collection.data_frequency)
        ibi_values = working_data['RR_list']/1000
        timestamps_ibi = np.cumsum(ibi_values) + bvp_collection.timestamps[0]
        data = np.stack((timestamps_ibi, ibi_values), axis=1)
        
        # Making sure that the timestamp drift between derived ibi and original BVP timestamps is not too large
        drift = bvp_collection.timestamps[-1] - timestamps_ibi[-1]
        
        if drift > BvpService.ACCEPTABLE_DRIFT:
            return Collection(bvp_collection.subject_id, np.zeros((0,2)), 0)
        
        #only keeping IBI values between 0.4 and 2
        mask = [BvpService.IBI_LOWER_BOUND < x < BvpService.IBI_UPPER_BOUND for x in ibi_values]
        data = data[mask]
        ibi_collection = Collection(bvp_collection.subject_id, data, 0)
        return ibi_collection
    
    @staticmethod
    def bvp_filter(signal):
        fH = 4
        fL = 0.5
        freq = 64
        nyquist = freq / 2
        sos = s.cheby2(4, 20, Wn=(fL / nyquist, fH / nyquist), btype='bandpass', output = 'sos')
        filtered = s.sosfiltfilt(sos, signal)
        return filtered

    @staticmethod
    def in_order(timestamps):
        for i in range(len(timestamps) - 1):
            if timestamps[i] > timestamps[i+1]:
                return False
        return True

            

        
