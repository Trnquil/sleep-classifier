import numpy as np
import pyedflib as pyedflib

from source import utils
from source.preprocessing.collection import Collection
from source.constants import Constants


class MesaPPGService(object):
    @staticmethod
    def load_raw(file_id):
        project_root = utils.get_project_root()

        edf_file = pyedflib.EdfReader(str(project_root.joinpath(Constants.MESA_DATA_PATH.joinpath('polysomnography/edfs/mesa-sleep-' + file_id + '.edf'))))
        signal_labels = edf_file.getSignalLabels()

        ppg_column = len(signal_labels) - 5

        sample_frequencies = edf_file.getSampleFrequencies()

        ppg = edf_file.readSignal(ppg_column)
        sf = sample_frequencies[ppg_column]

        time_ppg = np.array(range(0, len(ppg)))  # Get timestamps for heart rate data
        time_ppg = time_ppg / sf

        data = np.transpose(np.vstack((time_ppg, ppg)))
        data = utils.remove_nans(data)
        return Collection(subject_id=file_id, data=data, data_frequency=sf)
