import os


from source.constants import Constants
import source.utils as utils
from source.analysis.setup.feature_type import FeatureType
from source.data_services.dataset import DataSet

class PathService(object):
    filenames = {
        FeatureType.raw_hr.name: "HR.csv",
        FeatureType.raw_acc.name: "ACC.csv",
        FeatureType.raw_ibi.name: "IBI.csv",
        FeatureType.raw_bvp.name: "BVP.csv",
        
        FeatureType.cropped_count.name: "cropped_counts.out",
        FeatureType.cropped_ibi.name: "cropped_ibi.out",
        FeatureType.cropped_motion.name: "cropped_motion.out",
        FeatureType.cropped_hr.name: "cropped_hr.out",
        
        FeatureType.epoched_cluster.name: "clusters.out",
        FeatureType.epoched.name: "epoched_features.csv",
        FeatureType.epoched_hr.name: "epoched_features.csv",
        FeatureType.epoched_count.name: "epoched_features.csv",
        
        FeatureType.nightly.name: "nightly_features.csv"
        }
    
    
    @staticmethod
    def get_cropped_file_path(subject_id, session_id, feature_type):
        directory_path_string = str(subject_id) + "/" + str(session_id)
        
        return str(Constants.CROPPED_FILE_PATH.joinpath(directory_path_string)) + "/" + PathService.filenames[feature_type.name]
    
    @staticmethod
    def create_cropped_file_path(subject_id, session_id):
        directory_path_string = str(subject_id) + "/" + str(session_id)
        
        subject_folder_path = Constants.CROPPED_FILE_PATH.joinpath(str(subject_id))
        # creating a subject folder if it doesn't already exist
        if not os.path.exists(subject_folder_path):
            os.mkdir(subject_folder_path)
            
        sleep_session_path = Constants().CROPPED_FILE_PATH.joinpath(directory_path_string)
        # creating a sleep session folder if it doesn't already exist
        if not os.path.exists(sleep_session_path):
            os.mkdir(sleep_session_path)

    
    @staticmethod
    def get_nightly_file_path():
        return str(Constants.NIGHTLY_FILE_PATH) + "/" + PathService.filenames[FeatureType.nightly.name]
    
    @staticmethod
    def get_raw_file_paths(subject_id, feature_type):
        subject_dir = utils.get_project_root().joinpath('data/USI Sleep/E4_Data/' + subject_id)
        session_dirs = os.listdir(subject_dir)
        session_dirs.sort()
        
        #Removing .DS_Store from the list of directories because we don't care about it
        session_dirs.remove('.DS_Store')
        
        session_dirs = [str(subject_dir.joinpath(session_dirs[i])) + "/" + PathService.filenames[feature_type.name] 
         for i in range(len(session_dirs))]
     
        return [session_dirs[3]]
    
    @staticmethod
    def get_epoched_file_path(subject_id, session_id, feature_type, dataset):
        if(dataset.name == DataSet.usi.name):
            directory_path = Constants.EPOCHED_FILE_PATH.joinpath(Constants.USI_FOLDER_NAME)
        elif(dataset.name == DataSet.mesa.name):
            directory_path = Constants.EPOCHED_FILE_PATH.joinpath(Constants.MESA_FOLDER_NAME)
            
        full_path = directory_path.joinpath(subject_id + "/" + str(session_id))
        return str(full_path) + "/" + PathService.filenames[feature_type.name]
    
    @staticmethod
    def create_epoched_folder_path(subject_id, session_id, dataset):
        if(dataset.name == DataSet.usi.name):
            directory_path = Constants.EPOCHED_FILE_PATH.joinpath(Constants.USI_FOLDER_NAME)
        elif(dataset.name == DataSet.mesa.name):
            directory_path = Constants.EPOCHED_FILE_PATH.joinpath(Constants.MESA_FOLDER_NAME)
        
        if not (os.path.exists(directory_path)):
            os.mkdir(directory_path)
        
        subject_path = directory_path.joinpath(subject_id)
        if not (os.path.exists(subject_path)):
            os.mkdir(subject_path)
        
        session_path = subject_path.joinpath(session_id)
        if not (os.path.exists(session_path)):
            os.mkdir(session_path)
    
    @staticmethod
    def get_model_path():
        models_dir = utils.get_project_root().joinpath('data/imported models')
        return str(models_dir) + "/kmeans.pkl"