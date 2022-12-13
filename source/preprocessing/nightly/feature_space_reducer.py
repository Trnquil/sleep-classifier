from sklearn.decomposition import PCA

import pandas as pd

class FeatureSpaceReducer(object):
    
    @staticmethod
    def PCA(nightly_dataframe):
        features_df = nightly_dataframe.drop(columns=['subject_id', 'session_id', 'sleep_quality']).dropna()
        pca = PCA(n_components=2)
        features = pca.fit_transform(features_df.to_numpy)
        reduced_features_df = pd.DataFrame(features)
        reduced_features_df = reduced_features_df.add_prefix('pca_')
        return reduced_features_df
        