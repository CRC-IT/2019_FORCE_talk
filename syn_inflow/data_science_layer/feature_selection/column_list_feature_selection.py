
from .abstract_feature_selector import AbstractFeatureSelector


class ColumnListFeatureSelector(AbstractFeatureSelector):

    @classmethod
    def select_features(cls, x, y):
        obj = cls()
        return obj.select_features_from_model(x, y)

    def select_features_from_model(self, x, y):
        x_select = self.select_features_in_test_set(x)
        return x_select
