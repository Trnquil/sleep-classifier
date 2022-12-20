import seaborn as sns

from source.analysis.setup.feature_type import FeatureType


class FeatureSetService(object):

    @staticmethod
    def get_label(feature_types):
        string = ""
        if len(feature_types) == 1:
            return str(feature_types[0].value)
        
        for i in range(len(feature_types)):
            if i < len(feature_types) - 2:
                string = string + str(feature_types[i].value) + ", "
            elif i == len(feature_types) - 2:
                string = string + str(feature_types[i].value)
            else:
                string = string + " and " + str(feature_types[i].value)
        return string


    @staticmethod
    def get_color(feature_set: [FeatureType]):
        if set(feature_set) == {FeatureType.nightly_cluster_kmeans_kmeans}:
            return sns.xkcd_rgb["denim blue"]
        if set(feature_set) == {FeatureType.nightly_ibi}:
            return sns.xkcd_rgb["yellow orange"]
        if set(feature_set) == {FeatureType.nightly_cluster_kmeans, FeatureType.nightly_ibi}:
            return sns.xkcd_rgb["medium green"]
        if set(feature_set) == {FeatureType.nightly_cluster_kmeans, FeatureType.nightly_ibi, FeatureType.nightly_count}:
            return sns.xkcd_rgb["medium pink"]
        if set(feature_set) == {FeatureType.nightly_cluster_kmeans, FeatureType.nightly_ibi_from_ppg}:
            return sns.xkcd_rgb["plum"]
        if set(feature_set) == {FeatureType.nightly_cluster_kmeans, FeatureType.nightly_ibi}:
            return sns.xkcd_rgb["greyish"]
        if set(feature_set) == {FeatureType.nightly_cluster_kmeans, FeatureType.nightly_hr}:
            return sns.xkcd_rgb["green"]
        else:
            return sns.xkcd_rgb["grey"]
