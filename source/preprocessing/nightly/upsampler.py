
import pandas as pd
import numpy as np
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE

class Upsampler(object):
    
    @staticmethod 
    def random_duplication_upsampling(nightly_dataframe):
        
        df_1 = nightly_dataframe[nightly_dataframe.sleep_quality==1]
        df_0 = nightly_dataframe[nightly_dataframe.sleep_quality==0]
        
        if df_1.shape[0] > df_0.shape[0]:
            df_majority = df_1
            df_minority = df_0
        else:
            df_majority = df_0
            df_minority = df_1
                                        
        df_minority_upsampled = resample(df_minority, 
        replace=True,     # sample with replacement
        n_samples=df_majority.shape[0],    # to match majority class
        random_state=123)
        
        nightly_dataframe = pd.concat([df_majority, df_minority_upsampled])
        return nightly_dataframe
    
    @staticmethod
    def smote_upsampling(nightly_dataframe):
        nightly_dataframe = nightly_dataframe.dropna()
        height = nightly_dataframe.shape[0]
        X = nightly_dataframe.drop(columns=['sleep_quality'])
        y = nightly_dataframe['sleep_quality']

        s_ss_df = X[['subject_id', 'session_id']]
        
        oversample = SMOTE()
        X, y = oversample.fit_resample(X.drop(columns=['subject_id', 'session_id']), y)
        resampled_height = X.shape[0]
        
        # We're calculating the count of new datapoints to label them "RESAMPLED"
        count_new = resampled_height - height
        s_ss_resampled_df = pd.DataFrame("RESAMPLED", index=np.arange(0, count_new, 1), columns=['subject_id'])
        s_ss_resampled_df['session_id'] = ["SS_" + str(x + 1).zfill(2) for x in np.arange(count_new)]
        
        # Concatenating the old sleep subjects and sleepsessions and the newly labeled "RESAMPLED"
        s_ss_full_df = pd.concat([s_ss_df, s_ss_resampled_df], axis = 0)
        s_ss_full_df = s_ss_full_df.reset_index(drop=True)
        nightly_oversampled = pd.concat([s_ss_full_df, X, y], axis = 1)
        
        return nightly_oversampled