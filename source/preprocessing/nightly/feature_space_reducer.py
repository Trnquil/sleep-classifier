from sklearn.decomposition import PCA
from source.runner_parameters import RunnerParameters

import pandas as pd

class FeatureSpaceReducer(object):
    
    @staticmethod
    def PCA(nightly_dataframe):
        nightly_dataframe = nightly_dataframe.reset_index(drop=True).dropna()
        features_df = nightly_dataframe.drop(columns=['subject_id', 'session_id', 'sleep_quality'])
        pca = PCA(n_components=RunnerParameters.PCA_COMPONENTS)
        features = pca.fit_transform(features_df.to_numpy())
        reduced_features_df = pd.DataFrame(features)
        reduced_features_df = reduced_features_df.add_prefix('reduced_')
        reduced_features_df = pd.concat([nightly_dataframe[['subject_id', 'session_id']], 
                                         reduced_features_df,
                                         nightly_dataframe['sleep_quality']], axis=1)
        return reduced_features_df
        