from source.analysis.setup.attributed_classifier import AttributedClassifier
from source.analysis.performance.raw_performance import RawPerformance

import numpy as np


class ClassifierSummary(object):
    def __init__(self, attributed_classifier: AttributedClassifier, performance_dictionary):
        self.attributed_classifier = attributed_classifier
        self.performance_dictionary = performance_dictionary
        
    def get_unified_performance_dictionary(self):
        unified_performance_dictionary = {}
        for raw_performances_key, raw_performances in self.performance_dictionary.items():

            unified_true_labels = []
            unified_class_probabilites = []
            i = 0
            for raw_performance in [raw_performances]:
                true_labels = np.expand_dims(raw_performance.true_labels, axis=1)
                class_probabilites = raw_performance.class_probabilities
                if i == 0:
                    unified_true_labels = true_labels
                    unified_class_probabilites = class_probabilites
                else:
                    unified_true_labels = np.concatenate((unified_true_labels, true_labels), axis=0)
                    unified_class_probabilites = np.concatenate((unified_class_probabilites, class_probabilites), axis=0)
                i += 1
            
            unified_performance_dictionary[raw_performances_key] = RawPerformance(unified_true_labels, unified_class_probabilites)
        return unified_performance_dictionary
            
        
        
        
            
        
