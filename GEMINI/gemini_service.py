import sys
sys.path.insert(1, "..")

from GEMINI.utils.config import create_config
from GEMINI.utils.train_utils import train_model
import logging

import numpy as np
import torch

from matplotlib import pyplot as plt
from GEMINI.utils.utils import build_model, export_csv_results,load_and_build_train_dataset,load_and_build_val_dataset

from sklearn import metrics
from source import utils

from pathlib import Path
import os

class Arguments(object):
    def __init__(self, config: str, model_name='', seed=1, no_csv=False):
        self.config = config
        self.model_name = model_name
        self.seed = seed
        self.no_csv = no_csv
        
class GeminiService(object):
    
    args = Arguments(config = utils.get_project_root().joinpath("GEMINI/config_experiments/mesa/lenet5/GEMINI_hellinger_OVO_true_CLUSTERS_10_VAT_1.yml"))
    model_path = utils.get_project_root().joinpath("GEMINI/results/mesa/lenet5")
    
    def get_model_dirs():
        model_dirs = [x for x in os.listdir(GeminiService.model_path) 
                      if os.path.isdir(os.path.join(GeminiService.model_path, x))
                      and os.path.exists(os.path.join(GeminiService.model_path, x + '/model.pt'))]
        model_dirs.sort()
        return model_dirs
        
    def model_exists():
        model_dirs = GeminiService.get_model_dirs()
        model_exists = len(model_dirs) > 0
        return model_exists
    
    def load_newest_model_path():
        model_dirs = GeminiService.get_model_dirs()
        newest_dir = model_dirs[-1]
        return os.path.join(GeminiService.model_path, newest_dir + '/model.pt')
    
    def load_model():  
        config=create_config(GeminiService.args)
        model=build_model(config)
        model.load_state_dict(torch.load(GeminiService.load_newest_model_path())['model_state_dict'])
        return model

    def build_model():
        
        logging.info('Looking for configuration files')
        config=create_config(GeminiService.args)
        
        logging.info('Loading training dataset')
        train_ds=load_and_build_train_dataset(config)
        logging.info('Loading validation dataset')
        val_ds=load_and_build_val_dataset(config)
        
        logging.info('Building the model')
        model=build_model(config)
        
        logging.info('Training model')
        train_metrics,val_metrics=train_model(config,model,train_ds,val_ds)
        
        print("\nGEMINI Model: \n")
        print(model)
        
        ##################### For testing only
        val_dl = torch.utils.data.DataLoader(val_ds, batch_size=500, shuffle=False)
        train_dl = torch.utils.data.DataLoader(train_ds, batch_size=500, shuffle=False)
        
        with torch.no_grad():   
            for j, x_batch in enumerate(train_dl):
                # separating features and labels into different tensors
                features = x_batch[0].numpy()
                labels = x_batch[1].numpy()
                predictions = model(torch.tensor(features))[0].detach().numpy()
                clusters = [x.argmax() for x in predictions]
                break
        ##################### For testing only
        
        if not config.no_csv:
            logging.info('Exporting csv results')
            logging.debug('Exporting training metrics')
            export_csv_results(config,train_metrics)
            if val_metrics is not None:
                logging.debug('Exporting validation metrics')
                export_csv_results(config,val_metrics,False)      
        
        logging.info('Finished building GEMINI Model')
        
    @staticmethod
    def make_confusion_matrix(labels, clusters):
        # Making a confusion matrix between predicted classes and class labels
        confusion_matrix = metrics.confusion_matrix(labels, clusters)
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix)
        cm_display.plot()
        cm_display.save_fig('confusion_matrix.png')
        plt.clf()