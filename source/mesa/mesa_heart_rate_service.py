import numpy as np
import pyedflib as pyedflib

from source import utils
from source.preprocessing.collection import Collection


class MesaHeartRateService(object):
    @staticmethod
    def load(file_id):
        project_root = str(utils.get_project_root())

        edf_file = pyedflib.EdfReader(project_root + '/data/mesa/polysomnography/edfs/mesa-sleep-' + file_id + '.edf')
        signal_labels = edf_file.getSignalLabels()

        hr_column = len(signal_labels) - 2

        sample_frequencies = edf_file.getSampleFrequencies()

        heart_rate = edf_file.readSignal(hr_column)
        sf = sample_frequencies[hr_column]

        time_hr = np.array(range(0, len(heart_rate)))  # Get timestamps for heart rate data
        time_hr = time_hr / sf

        data = np.transpose(np.vstack((time_hr, heart_rate)))
        data = utils.remove_nans(data)
        
        mask = [hr > 0 and hr != 100 for hr in data[:,1]]
        data = data[mask]
        
        return Collection(subject_id=file_id, data=data, data_frequency=sf)
