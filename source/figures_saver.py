from source.constants import Constants
from source.data_services.dataset import DataSet
import os

class FiguresSaver(object):
    
    def save_figures(dataset):
        settings_dirs = os.listdir(FiguresSaver.get_figures_path(dataset).joinpath(".."))
        
        i = 0
        while("settings " + str(i) in settings_dirs):
            i += 1
        
        src = FiguresSaver.get_figures_path(dataset)
        dst = FiguresSaver.get_figures_path(dataset).joinpath("..").joinpath("settings " + str(i))
        os.rename(src, dst)
        FiguresSaver.build_current_settings_dirs(dataset)
        
    def build_current_settings_dirs(dataset):
        os.mkdir(FiguresSaver.get_figures_path(dataset))
        os.mkdir(FiguresSaver.get_figures_path(dataset).joinpath("performance analysis"))
        os.mkdir(FiguresSaver.get_figures_path(dataset).joinpath("umap"))
        os.mkdir(FiguresSaver.get_figures_path(dataset).joinpath("clusters"))
        os.mkdir(FiguresSaver.get_figures_path(dataset).joinpath("cluster analysis"))
        
    def get_figures_path(dataset):
        if(dataset.name == DataSet.usi.name):     
            path = Constants.FIGURE_FILE_PATH.joinpath(Constants.USI_FOLDER_NAME).joinpath("current settings")
        elif(dataset.name == DataSet.mss.name):
            path = Constants.FIGURE_FILE_PATH.joinpath(Constants.MSS_FOLDER_NAME).joinpath("current settings")
        return path
        