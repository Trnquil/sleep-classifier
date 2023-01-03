import csv

import numpy as np
import pandas as pd

from source import utils
from source.preprocessing.collection import Collection
from source.constants import Constants


class MesaActigraphyService(object):
    @staticmethod
    def load_raw(file_id):
        line_align = -1  # Find alignment line between PSG and actigraphy
        project_root = utils.get_project_root()

        with open(str(project_root.joinpath(Constants.MESA_DATA_PATH.joinpath('overlap/mesa-actigraphy-psg-overlap.csv')))) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            next(csv_file)
            for row in csv_reader:
                if int(row[0]) == int(file_id):
                    line_align = int(row[1])

        activity = []
        elapsed_time_counter = 0

        if line_align == -1:  # If there was no alignment found
            return Collection(subject_id=file_id, data=np.array([[-1], [-1]]), data_frequency = 0)

        with open(str(project_root.joinpath(Constants.MESA_DATA_PATH.joinpath('actigraphy/mesa-sleep-' + file_id + '.csv')))) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            next(csv_file)
            for row in csv_reader:
                if int(row[1]) >= line_align:
                    if row[4] == '':
                        activity.append([elapsed_time_counter, np.nan])
                    else:
                        activity.append([elapsed_time_counter, float(row[4])])
                    elapsed_time_counter = elapsed_time_counter + 30

        data = np.array(activity)
        data = utils.remove_nans(data)
        
        # Eventually want to return a dataframe, not a collection
        # count_dataframe = pd.DataFrame(data)
        # count_dataframe.columns = ["epoch_timestamp", "count"]
        return Collection(file_id, data, 0)
