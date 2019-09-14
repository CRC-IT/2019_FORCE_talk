
from .abstract_feature_selector import AbstractFeatureSelector
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2


class PercentileClassifierSelector(AbstractFeatureSelector):

    @classmethod
    def select_features(cls, x, y):
        obj = cls()
        return obj.select_features_from_model(x, y)

    def select_features_from_model(self, x, y, percentile=10):
        score_func = chi2

        selector = SelectPercentile(score_func=score_func, percentile=percentile)
        selector.fit_transform(x, y)
        features = selector.get_support(indices=True)
        self.best_features = [column for column in x.columns[features]]
        x_select = self.select_features_in_test_set(x)

        return x_select
