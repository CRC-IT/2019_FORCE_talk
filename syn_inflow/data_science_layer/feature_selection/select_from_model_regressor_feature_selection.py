
from .abstract_feature_selector import AbstractFeatureSelector
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso

class SelectFromModelRegressorSelector(AbstractFeatureSelector):

    @classmethod
    def select_features(cls, x, y):
        obj = cls()
        return obj.select_features_from_model(x, y)

    def select_features_from_model(self, x, y):

        selector = SelectFromModel(estimator=Lasso().fit(x, y), threshold=self.threshold, prefit=self.prefit,
                                   norm_order=self.norm_order, max_features=self.max_features)
        selector.fit_transform(x, y)
        features = selector.get_support(indices=True)
        self.best_features = [column for column in x.columns[features]]
        x_select = self.select_features_in_test_set(x)

        return x_select

