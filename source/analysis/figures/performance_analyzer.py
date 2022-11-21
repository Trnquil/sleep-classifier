from sklearn import metrics
from source.constants import Constants
from source.analysis.setup.feature_set_service import FeatureSetService
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

class PerformanceAnalyzer(object):
    
    @staticmethod
    def make_overall_performance_summary(classifier_summary):
        
        performance_dictionary = classifier_summary.get_unified_performance_dictionary()
        for feature_set in performance_dictionary:
            raw_performance = performance_dictionary[feature_set]
            true_labels = raw_performance.true_labels
            class_probabilities = raw_performance.class_probabilities
            predicted_labels = [round(x) for x in class_probabilities[:,1]]
            metrics_string = metrics.classification_report(true_labels, predicted_labels)
            
            text_file = open(str(Constants.ANALYSIS_FILE_PATH) + '/' + FeatureSetService.get_label(feature_set) + '_' \
                              + classifier_summary.attributed_classifier.name + '_overall_performance.txt', "w")
            text_file.write(metrics_string)
            text_file.close()
            




