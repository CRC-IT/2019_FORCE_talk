
from .abstract_feature_selector import AbstractFeatureSelector
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression


class KBestRegressorSelector(AbstractFeatureSelector):

    @classmethod
    def select_features(cls, x, y):
        obj = cls()
        return obj.select_features_from_model(x, y)

    def select_features_from_model(self, x, y, k=10):
        score_func = f_regression

        selector = SelectKBest(score_func=score_func, k=k)
        selector.fit_transform(x, y)
        features = selector.get_support(indices=True)
        self.best_features = [column for column in x.columns[features]]
        x_select = self.select_features_in_test_set(x)

        return x_select
