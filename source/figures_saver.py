from source.constants import Constants
import os

class FiguresSaver(object):
    
    def save_figures():
        settings_dirs = os.listdir(Constants.FIGURE_FILE_PATH.joinpath(".."))
        
        i = 0
        while("settings " + str(i) in settings_dirs):
            i += 1
        
        src = Constants.FIGURE_FILE_PATH
        dst = Constants.FIGURE_FILE_PATH.joinpath("..").joinpath("settings " + str(i))
        os.rename(src, dst)
        FiguresSaver.build_current_settings_dirs()
        
    def build_current_settings_dirs():
        os.mkdir(Constants.FIGURE_FILE_PATH)
        os.mkdir(Constants.ANALYSIS_FILE_PATH)
        os.mkdir(Constants.FIGURE_FILE_PATH.joinpath("umap"))
        os.mkdir(Constants.FIGURE_FILE_PATH.joinpath("clusters"))
        os.mkdir(Constants.FIGURE_FILE_PATH.joinpath("cluster analysis"))