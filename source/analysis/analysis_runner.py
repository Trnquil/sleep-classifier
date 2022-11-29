import sys
sys.path.insert(1, '../..')

import time

from sklearn.neural_network import MLPClassifier

from source import utils
from source.analysis.classification.classifier_summary_builder import SleepWakeClassifierSummaryBuilder, \
    ThreeClassClassifierSummaryBuilder
from source.analysis.figures.curve_plot_builder import CurvePlotBuilder
from source.analysis.figures.performance_plot_builder import PerformancePlotBuilder
from source.analysis.setup.attributed_classifier import AttributedClassifier
from source.analysis.setup.feature_type import FeatureType
from source.analysis.setup.subject_builder import SubjectBuilder
from source.analysis.tables.table_builder import TableBuilder
from source.constants import Constants
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from source.analysis.clustering.clustering_analyzer import ClusterAnalyzer
from source.data_services.data_service import DataService
from source.data_services.data_loader import DataLoader
from source.data_services.dataset import DataSet
from source.analysis.figures.performance_analyzer import PerformanceAnalyzer
from source.preprocessing.clustering.clustering_feature_service import ClusteringFeatureService
from source.data_services.data_frame_loader import DataFrameLoader


import pandas as pd

def cluster_analysis():
    
    print("Running Cluster Analysis...")
    
    feature_types = ClusteringFeatureService.cluster_feature_types
    feature_types.append(FeatureType.epoched_cluster)
    feature_types.append(FeatureType.epoched_sleep_label)
    
    full_df = DataFrameLoader.load_feature_dataframe(feature_types, DataSet.mesa)
    
    features_df = full_df.drop(columns=["epoch_timestamp", "cluster", "sleep_label"])
    clusters_df = full_df["cluster"]
    labels_df = full_df["sleep_label"]
    
    ClusterAnalyzer.analyze(features_df, clusters_df, labels_df)
    
    
def figures_leave_one_out():
    attributed_classifiers = [AttributedClassifier(name='Nearest Neighbors',
                                                 classifier=KNeighborsClassifier()),
                              AttributedClassifier(name='Random Forest',
                                                 classifier=RandomForestClassifier(max_depth=10, random_state=0)),
                              AttributedClassifier(name='SVM',
                                                 classifier=SVC(probability=True))]

    feature_sets = [[FeatureType.nightly_cluster, FeatureType.nightly_ibi]]
    
    for attributed_classifier in attributed_classifiers:
        if Constants.VERBOSE:
            print('Running ' + attributed_classifier.name + '...')
        classifier_summary = SleepWakeClassifierSummaryBuilder.build_leave_one_out(attributed_classifier, feature_sets)

        PerformanceAnalyzer.make_overall_performance_summary(classifier_summary)
        PerformancePlotBuilder.make_histogram_with_thresholds(classifier_summary)
        PerformancePlotBuilder.make_single_threshold_histograms(classifier_summary)
        CurvePlotBuilder.make_roc_sw(classifier_summary)
        CurvePlotBuilder.make_pr_sw(classifier_summary)
        TableBuilder.print_table_sw(classifier_summary)
    
    
def figures_leave_one_out_sleep_wake_performance():
    attributed_classifier = AttributedClassifier(name='Neural Net',
                                                 classifier=MLPClassifier(activation='relu',
                                                                          hidden_layer_sizes=(15, 15, 15),
                                                                          max_iter=1000, alpha=0.01, solver='lbfgs'))

    feature_sets = [[FeatureType.count, FeatureType.heart_rate, FeatureType.circadian_model]]

    if Constants.VERBOSE:
        print('Running ' + attributed_classifier.name + '...')
    classifier_summary = SleepWakeClassifierSummaryBuilder.build_leave_one_out(attributed_classifier, feature_sets)
    PerformancePlotBuilder.make_histogram_with_thresholds(classifier_summary)
    PerformancePlotBuilder.make_single_threshold_histograms(classifier_summary)


def figures_leave_one_out_three_class_performance():
    attributed_classifier = AttributedClassifier(name='Neural Net',
                                                 classifier=MLPClassifier(activation='relu',
                                                                          hidden_layer_sizes=(15, 15, 15),
                                                                          max_iter=1000, alpha=0.01, solver='lbfgs'))

    feature_sets = utils.get_base_feature_sets()

    if Constants.VERBOSE:
        print('Running ' + attributed_classifier.name + '...')
    classifier_summary = ThreeClassClassifierSummaryBuilder.build_leave_one_out(attributed_classifier, feature_sets)
    PerformancePlotBuilder.make_bland_altman(classifier_summary)


def figure_leave_one_out_roc_and_pr():
    classifiers = utils.get_classifiers()
    feature_sets = utils.get_base_feature_sets()

    for attributed_classifier in classifiers:
        if Constants.VERBOSE:
            print('Running ' + attributed_classifier.name + '...')
        classifier_summary = SleepWakeClassifierSummaryBuilder.build_leave_one_out(attributed_classifier, feature_sets)

        CurvePlotBuilder.make_roc_sw(classifier_summary)
        CurvePlotBuilder.make_pr_sw(classifier_summary)
        TableBuilder.print_table_sw(classifier_summary)

    CurvePlotBuilder.combine_plots_as_grid(classifiers, len(SubjectBuilder.get_all_subject_ids()), '_sw_pr')
    CurvePlotBuilder.combine_plots_as_grid(classifiers, len(SubjectBuilder.get_all_subject_ids()), '_sw_roc')


def figures_mc_sleep_wake():
    classifiers = utils.get_classifiers()

    feature_sets = utils.get_base_feature_sets()
    trial_count = 20

    for attributed_classifier in classifiers:
        if Constants.VERBOSE:
            print('Running ' + attributed_classifier.name + '...')
        classifier_summary = SleepWakeClassifierSummaryBuilder.build_monte_carlo(attributed_classifier, feature_sets,
                                                                                 trial_count)

        CurvePlotBuilder.make_roc_sw(classifier_summary)
        CurvePlotBuilder.make_pr_sw(classifier_summary)
        TableBuilder.print_table_sw(classifier_summary)

    CurvePlotBuilder.combine_plots_as_grid(classifiers, trial_count, '_sw_pr')
    CurvePlotBuilder.combine_plots_as_grid(classifiers, trial_count, '_sw_roc')


def figures_mc_three_class():
    classifiers = utils.get_classifiers()
    feature_sets = utils.get_base_feature_sets()
    trial_count = 20

    three_class_performance_summaries = []
    for attributed_classifier in classifiers:
        if Constants.VERBOSE:
            print('Running ' + attributed_classifier.name + '...')
        classifier_summary = ThreeClassClassifierSummaryBuilder.build_monte_carlo(attributed_classifier, feature_sets,
                                                                                  trial_count)

        CurvePlotBuilder.make_roc_one_vs_rest(classifier_summary)
        three_class_performance_dictionary = CurvePlotBuilder.make_three_class_roc(classifier_summary)

        classifier_summary.performance_dictionary = three_class_performance_dictionary
        three_class_performance_summaries.append(classifier_summary)

    TableBuilder.print_table_three_class(three_class_performance_summaries)
    CurvePlotBuilder.combine_plots_as_grid(classifiers, trial_count, '_three_class_roc')
    CurvePlotBuilder.combine_plots_as_grid(classifiers, trial_count, '_ovr_rem_roc')
    CurvePlotBuilder.combine_plots_as_grid(classifiers, trial_count, '_ovr_nrem_roc')
    CurvePlotBuilder.combine_plots_as_grid(classifiers, trial_count, '_ovr_wake_roc')


def figures_mesa_sleep_wake():
    classifiers = utils.get_classifiers()
    # Uncomment to just use MLP:
    # classifiers = [AttributedClassifier(name='Neural Net',
    #                                     classifier=MLPClassifier(activation='relu', hidden_layer_sizes=(15, 15, 15),
    #                                                              max_iter=1000, alpha=0.01, solver='lbfgs'))]

    feature_sets = utils.get_base_feature_sets()

    for attributed_classifier in classifiers:
        if Constants.VERBOSE:
            print('Running ' + attributed_classifier.name + '...')
        classifier_summary = SleepWakeClassifierSummaryBuilder.build_mesa(attributed_classifier, feature_sets)
        CurvePlotBuilder.make_roc_sw(classifier_summary, '_mesa')
        CurvePlotBuilder.make_pr_sw(classifier_summary, '_mesa')
        TableBuilder.print_table_sw(classifier_summary)

    CurvePlotBuilder.combine_plots_as_grid(classifiers, 1, '_mesa_sw_pr')
    CurvePlotBuilder.combine_plots_as_grid(classifiers, 1, '_mesa_sw_roc')


def figures_mesa_three_class():
    classifiers = utils.get_classifiers()

    # Uncomment to just use MLP:
    # classifiers = [AttributedClassifier(name='Neural Net', classifier=MLPClassifier(activation='relu', hidden_layer_sizes=(15, 15, 15),
    #                                                            max_iter=1000, alpha=0.01, solver='lbfgs'))]

    feature_sets = utils.get_base_feature_sets()
    three_class_performance_summaries = []

    for attributed_classifier in classifiers:
        if Constants.VERBOSE:
            print('Running ' + attributed_classifier.name + '...')
        classifier_summary = ThreeClassClassifierSummaryBuilder.build_mesa_leave_one_out(attributed_classifier,
                                                                                         feature_sets)
        PerformancePlotBuilder.make_bland_altman(classifier_summary, '_mesa')
        PerformancePlotBuilder.make_single_threshold_histograms(classifier_summary, '_mesa')

    for attributed_classifier in classifiers:
        if Constants.VERBOSE:
            print('Running ' + attributed_classifier.name + '...')
        classifier_summary = ThreeClassClassifierSummaryBuilder.build_mesa_all_combined(attributed_classifier,
                                                                                        feature_sets)
        three_class_performance_dictionary = CurvePlotBuilder.make_three_class_roc(classifier_summary, '_mesa')
        classifier_summary.performance_dictionary = three_class_performance_dictionary
        three_class_performance_summaries.append(classifier_summary)
        CurvePlotBuilder.combine_sw_and_three_class_plots(attributed_classifier, 1, 'mesa')

    TableBuilder.print_table_three_class(three_class_performance_summaries)
    CurvePlotBuilder.combine_plots_as_grid(classifiers, 1, '_mesa_three_class_roc')


def figures_compare_time_based_features():
    classifiers = utils.get_classifiers()
    feature_sets = [[FeatureType.count, FeatureType.heart_rate],
                    [FeatureType.count, FeatureType.heart_rate, FeatureType.time],
                    [FeatureType.count, FeatureType.heart_rate, FeatureType.cosine],
                    [FeatureType.count, FeatureType.heart_rate, FeatureType.circadian_model]]

    trial_count = 50

    for attributed_classifier in classifiers:
        if Constants.VERBOSE:
            print('Running ' + attributed_classifier.name + '...')
        classifier_summary = SleepWakeClassifierSummaryBuilder.build_monte_carlo(attributed_classifier, feature_sets,
                                                                                 trial_count)

        CurvePlotBuilder.make_roc_sw(classifier_summary, '_time_only')
        CurvePlotBuilder.make_pr_sw(classifier_summary, '_time_only')
        TableBuilder.print_table_sw(classifier_summary)

    CurvePlotBuilder.combine_plots_as_grid(classifiers, trial_count, '_time_only_sw_pr')
    CurvePlotBuilder.combine_plots_as_grid(classifiers, trial_count, '_time_only_sw_roc')


if __name__ == "__main__":
    start_time = time.time()
    figures_leave_one_out()
    cluster_analysis()
    #figure_leave_one_out_roc_and_pr()
    #
    # figures_mc_sleep_wake()
    # figures_mc_three_class()
    #
    # figures_leave_one_out_sleep_wake_performance()
    # figures_leave_one_out_three_class_performance()
    #
    # figures_mesa_sleep_wake()
    # figures_mesa_three_class()
    #
    # figures_compare_time_based_features()
    end_time = time.time()

    print('Elapsed time to generate figure: ' + str((end_time - start_time) / 60) + ' minutes')
