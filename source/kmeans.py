#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed Oct  5 14:41:20 2022

@author: Julien Wolfensberger
"""

import sys
# caution: path[0] is reserved for script path (or '' in REPL)
# We are inserting the sleepclassifiers as a module 
sys.path.insert(1, '/Users/julien/OneDrive/ETH/HS22/Bachelor Thesis/sleepclassifiers')

from source.analysis.classification.classifier_input_builder import ClassifierInputBuilder
from sklearn.cluster import KMeans
from source.analysis.setup.subject_builder import SubjectBuilder
from source.analysis.setup.feature_type import FeatureType

def train(subject_ids, subject_dictionary, feature_set: [FeatureType]):
    
    
    # Let us get labels as well as features
    training_x, training_y = ClassifierInputBuilder.get_array(subject_ids, subject_dictionary, feature_set)

    # Here we decide on what classifier that we should be using. We decide for Kmeans in the beginning
    classifier=KMeans(n_clusters=5, random_state=0)

    # We are now fitting our features to the cluster
    classifier.fit(training_x)

    #Here we do some predictions. We can then check how similar y and x are.
    class_predictions = classifier.predict(training_x)
        
    

# Here we are getting all subject IDs of subjects participating in the study
subject_ids = SubjectBuilder.get_all_subject_ids()

# Next we build a dictionary of subjects so that we can more easily access their attributes
subject_dictionary = SubjectBuilder.get_subject_dictionary()

# With this line of code we decide what features to use for our classification
feature_set = [FeatureType.count, FeatureType.heart_rate, FeatureType.cosine]

train(subject_ids, subject_dictionary, feature_set)
    
    
