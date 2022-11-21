from sklearn import metrics
from source.constants import Constants
from source.analysis.setup.feature_set_service import FeatureSetService
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

class PerformanceAnalyzer(object):
    
    @staticmethod
    def make_overall_performance_summary(classifier_summary):
        
        performance_dictionary = classifier_summary.get_unified_performance_dictionary()
        for feature_set in performance_dictionary:
            raw_performance = performance_dictionary[feature_set]
            y_true = raw_performance.true_labels
            class_probabilities = raw_performance.class_probabilities
            y_pred = [round(x) for x in class_probabilities[:,1]]
            
            decimals = 2
            performance_metrics = {}
            performance_metrics['accuracy'] = round(metrics.accuracy_score(y_true, y_pred), decimals)
            performance_metrics['precision'] = round(metrics.precision_score(y_true, y_pred), decimals)
            performance_metrics['recall'] = round(metrics.recall_score(y_true, y_pred), decimals)
            performance_metrics['f1'] = round(metrics.f1_score(y_true, y_pred), decimals)
            performance_metrics['balanced_accuracy'] = round(metrics.balanced_accuracy_score(y_true, y_pred), decimals)
            performance_metrics['roc_auc_score'] = round(metrics.roc_auc_score(y_true, y_pred), decimals)
            
            
            x = list(performance_metrics.keys())
            y = list(performance_metrics.values())
            fig, ax = plt.subplots()    
            width = 0.75 # the width of the bars 
            ind = np.arange(len(x))  # the x locations for the groups
            ax.barh(ind, y, width, color="skyblue")
            ax.set_yticks(ind)
            ax.set_xlim((0,1.1))
            ax.set_yticklabels(x, minor=False)
            plt.title('Performance Metrics') 
            plt.tight_layout()
            
            for bars in ax.containers:
                ax.bar_label(bars, padding=3)
            plt.savefig(str(Constants.ANALYSIS_FILE_PATH) + '/' + FeatureSetService.get_label(feature_set) + '_' \
                              + classifier_summary.attributed_classifier.name + '_overall_performance.png')
            
            plt.clf()






