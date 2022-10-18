import os


from source.constants import Constants
import source.utils as utils


class PathService(object):
    
    @staticmethod
    def get_cropped_folder_path(subject_id, session_id):
        directory_path_string = str(subject_id) + "/" + str(session_id)
        
        subject_folder_path = Constants().CROPPED_FILE_PATH.joinpath(str(subject_id))
        # creating a subject folder if it doesn't already exist
        if not os.path.exists(subject_folder_path):
            os.mkdir(subject_folder_path)
            
        sleep_session_path = Constants().CROPPED_FILE_PATH.joinpath(directory_path_string)
        # creating a sleep session folder if it doesn't already exist
        if not os.path.exists(sleep_session_path):
            os.mkdir(sleep_session_path)
        
        return str(Constants.CROPPED_FILE_PATH.joinpath(directory_path_string))
    
    @staticmethod
    def get_raw_folder_path(subject_id):
        subject_dir = utils.get_project_root().joinpath('data/USI Sleep/E4_Data/' + subject_id)
        session_dirs = os.listdir(subject_dir)
        session_dirs.sort()
        
        #Removing .DS_Store from the list of directories because we don't care about it
        session_dirs.remove('.DS_Store')
        
        #For now we are simply returning the first session
        #TODO: Return all directories, not only the first one
        return str(subject_dir.joinpath(session_dirs[2]))
    
    @staticmethod
    def get_feature_folder_path(subject_id, session_id):
        directory_path_string = Constants.FEATURE_FILE_PATH.joinpath(subject_id + "/" + str(session_id))
        
        if not (os.path.exists(Constants.FEATURE_FILE_PATH.joinpath(subject_id))):
            os.mkdir(Constants.FEATURE_FILE_PATH.joinpath(subject_id))
        
        if not (os.path.exists(directory_path_string)):
            os.mkdir(directory_path_string)
        return str(Constants.FEATURE_FILE_PATH.joinpath(directory_path_string))