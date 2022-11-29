import seaborn as sns

from source.analysis.setup.feature_type import FeatureType


class FeatureSetService(object):

    @staticmethod
    def get_label(feature_set: [FeatureType]):
        if set(feature_set) == {FeatureType.nightly_cluster}:
            return 'Clusters only'
        if set(feature_set) == {FeatureType.nightly_ibi}:
            return 'IBI only'
        if set(feature_set) == {FeatureType.nightly_cluster, FeatureType.nightly_ibi}:
            return 'Clusters and IBI'
        if set(feature_set) == {FeatureType.nightly_cluster, FeatureType.nightly_count}:
            return 'Clusters and Count'
        if set(feature_set) == {FeatureType.nightly_ibi, FeatureType.nightly_count}:
            return 'IBI and Count'
        if set(feature_set) == {FeatureType.nightly_cluster, FeatureType.nightly_ibi, FeatureType.nightly_count}:
            return 'Clusters, IBI and Count'
        if set(feature_set) == {FeatureType.nightly_cluster, FeatureType.nightly_hr}:
            return 'Clusters and HR'
        else:
            return 'unknown feature set'

    @staticmethod
    def get_color(feature_set: [FeatureType]):
        if set(feature_set) == {FeatureType.nightly_cluster}:
            return sns.xkcd_rgb["denim blue"]
        if set(feature_set) == {FeatureType.nightly_ibi}:
            return sns.xkcd_rgb["yellow orange"]
        if set(feature_set) == {FeatureType.nightly_cluster, FeatureType.nightly_ibi}:
            return sns.xkcd_rgb["medium green"]
        if set(feature_set) == {FeatureType.nightly_cluster, FeatureType.nightly_ibi, FeatureType.nightly_count}:
            return sns.xkcd_rgb["medium pink"]
        if set(feature_set) == {FeatureType.nightly_cluster, FeatureType.nightly_count}:
            return sns.xkcd_rgb["plum"]
        if set(feature_set) == {FeatureType.nightly_ibi, FeatureType.nightly_count}:
            return sns.xkcd_rgb["greyish"]
        if set(feature_set) == {FeatureType.nightly_cluster, FeatureType.nightly_hr}:
            return sns.xkcd_rgb["green"]
        else:
            return sns.xkcd_rgb["grey"]
