import pandas as pd
import os
from source.constants import Constants

class ExceptionLogger(object):
    
    file_path = Constants.LOGS_FILE_PATH.joinpath("exception_logs.csv")
    
    def append_exception(subject_id, session_id, feature_type, dataset):
        if not os.path.exists(Constants.LOGS_FILE_PATH):
            os.mkdir(Constants.LOGS_FILE_PATH)
           
        exception_dict = {
            'subject_id': [subject_id],
            'session_id': [session_id],
            'feature_type': [feature_type],
            'dataset': [dataset]
            }
        
        exception_df = pd.DataFrame.from_dict(exception_dict)
        
        if (os.path.exists(ExceptionLogger.file_path)):
            exceptions_df = pd.read_csv(ExceptionLogger.file_path)
            exceptions_df = pd.concat([exceptions_df, exception_df], axis=0)
        else:
            exceptions_df = exception_df
            
        
        exceptions_df.to_csv(ExceptionLogger.file_path, index=False)
        
        
    def remove_logs():
        if os.path.exists(ExceptionLogger.file_path):
            os.remove(ExceptionLogger.file_path)