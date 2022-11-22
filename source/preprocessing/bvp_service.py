import numpy as np

from source.preprocessing.collection import Collection

import heartpy as hp
import scipy.signal as s


class BvpService(object):
    IBI_LOWER_BOUND = 0.4
    IBI_UPPER_BOUND = 2
    
    @staticmethod
    def get_ibi_from_bvp(bvp_collection):
        # Working on BVP values to produce IBI sequence
        bvp_values = bvp_collection.values.squeeze()
        filtered = BvpService.bvp_filter(bvp_values)
        working_data, measures = hp.process(filtered, bvp_collection.data_frequency)
        ibi_values = working_data['RR_list']/1000
        timestamps_ibi = np.cumsum(ibi_values) + bvp_collection.timestamps[0]
        data = np.stack((timestamps_ibi, ibi_values), axis=1)
        
        # Making sure that the timestamp drift between derived ibi and original BVP timestamps is not too large
        drift = bvp_collection.timestamps[-1] - timestamps_ibi[-1] 
        if drift > 10:
            print("Warning: Ibi Drift over 10 seconds, sleep session won't be included. Drift: " + str(drift))
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
