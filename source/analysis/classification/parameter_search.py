from sklearn.model_selection import GridSearchCV


class ParameterSearch(object):
    parameter_dictionary = {
        'SVM': {'gamma': [0.001, 0.01 ,0.1, 1, 10, 100, 1000] , 'C': [0.001, 0.01, 0.1, 1, 10, 100]},
        'Logistic Regression': {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2']},
        'Random Forest': {'max_depth': [2, 5, 10, 50, 100, 1000]},
        'k-Nearest Neighbors': {'n_neighbors': [3, 5, 7, 11, 13, 15, 20, 30, 50], 'weights': ['distance']},
        'Neural Net': {'alpha': [0.1, 0.01, 0.001, 0.0001, 0.00001]}
    }

    @staticmethod
    def run_search(attributed_classifier, training_x, training_y, scoring):
        parameter_range = ParameterSearch.parameter_dictionary[attributed_classifier.name]
        grid_search = GridSearchCV(attributed_classifier.classifier, parameter_range, scoring=scoring, cv=3)
        grid_search.fit(training_x, training_y)
        return grid_search.best_params_
