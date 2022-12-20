import sys
sys.path.insert(1, '..')

import time

from source.preprocessing.preprocessing import Preprocessing
from source.analysis.analysis import Analysis
from source.exception_logger import ExceptionLogger
from source.runner_parameters import RunnerParameters
from source.figures_saver import FiguresSaver
from source.data_services.dataset import DataSet


def run():

    run_preprocessing()
    
    # run_analysis()

    
def run_preprocessing():
    
    start_time = time.time()
        
    ExceptionLogger.remove_logs()
    
    # Preprocessing.build_usi_cropped()
    # Preprocessing.build_mss_cropped()
    
    # Preprocessing.build_usi_epoched()
    # Preprocessing.build_mss_epoched()
    # Preprocessing.build_mesa_epoched()
    
    Preprocessing.build_cluster_features_usi()
    Preprocessing.build_cluster_features_mesa()
    Preprocessing.build_cluster_features_mss()
    
    Preprocessing.build_clusters()
    
    Preprocessing.build_nightly_usi()
    Preprocessing.build_nightly_mss()
    
    end_time = time.time()
    
    print("Execution took " + str((end_time - start_time) / 60) + " minutes")
    
def run_analysis():
    if __name__ == "__main__":
        start_time = time.time()
        
        Analysis.all_figures(DataSet.usi)
        Analysis.all_figures(DataSet.mss)
        # Analysis.cluster_analysis()
        
        RunnerParameters.print_settings(DataSet.usi)
        RunnerParameters.print_settings(DataSet.mss)
        
        FiguresSaver.save_figures(DataSet.usi)
        FiguresSaver.save_figures(DataSet.mss)

        end_time = time.time()

        print('Elapsed time to generate figures: ' + str((end_time - start_time) / 60) + ' minutes')
    
    
run()