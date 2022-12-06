import sys
sys.path.insert(1, '..')

import time

from source.preprocessing.preprocessing import Preprocessing
from source.analysis.analysis import Analysis
from source.data_services.data_frame_loader import DataFrameLoader
from source.exception_logger import ExceptionLogger

def run():
    
    run_preprocessing()
    
    run_analysis()
    
    
def run_preprocessing():
    
    start_time = time.time()
        
    ExceptionLogger.remove_logs()
    Preprocessing.build_cropped()
    
    Preprocessing.build_epoched()
    Preprocessing.build_mesa_epoched()
    
    Preprocessing.build_cluster_features()
    Preprocessing.build_cluster_features_mesa()
    
    Preprocessing.build_clusters()
    
    Preprocessing.build_nightly()
    
    end_time = time.time()
    
    print("Execution took " + str((end_time - start_time) / 60) + " minutes")
    
def run_analysis():
    if __name__ == "__main__":
        start_time = time.time()
        
        Analysis.all_figures()
        # Analysis.cluster_analysis()

        end_time = time.time()

        print('Elapsed time to generate figures: ' + str((end_time - start_time) / 60) + ' minutes')
        
run()