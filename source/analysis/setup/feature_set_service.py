import seaborn as sns

from source.analysis.setup.feature_type import FeatureType


class FeatureSetService(object):

    @staticmethod
    def get_label(feature_set: [FeatureType]):
        if set(feature_set) == {FeatureType.nightly_cluster}:
            return 'Clusters only'
        if set(feature_set) == {FeatureType.nightly_hr}:
            return 'HR only'
        if set(feature_set) == {FeatureType.nightly_cluster, FeatureType.nightly_hr}:
            return 'Clusters and HR'

    @staticmethod
    def get_color(feature_set: [FeatureType]):
        if set(feature_set) == {FeatureType.nightly_cluster}:
            return sns.xkcd_rgb["denim blue"]
        if set(feature_set) == {FeatureType.nightly_hr}:
            return sns.xkcd_rgb["yellow orange"]
        if set(feature_set) == {FeatureType.nightly_cluster, FeatureType.nightly_hr}:
            return sns.xkcd_rgb["medium green"]
        # if set(feature_set) == {FeatureType.count, FeatureType.heart_rate, FeatureType.circadian_model}:
        #     return sns.xkcd_rgb["medium pink"]
        # if set(feature_set) == {FeatureType.count, FeatureType.heart_rate, FeatureType.cosine}:
        #     return sns.xkcd_rgb["plum"]
        # if set(feature_set) == {FeatureType.count, FeatureType.heart_rate, FeatureType.time}:
        #     return sns.xkcd_rgb["greyish"]
