
from .abstract_feature_selector import AbstractFeatureSelector
from ..machine_learning.extra_trees_classifier import ExtraTreesClassifierModel
import pandas as pd
import numpy as np


class ExtraTreesClassifierFeatureSelector(AbstractFeatureSelector):

    @classmethod
    def select_features(cls, x, y):
        obj = cls()
        return obj.select_features_from_model(x, y)

    def select_features_from_model(self, x, y):
        x['RANDOM_NUMBER'] = np.random.normal(0, 1, x.shape[0])

        et = ExtraTreesClassifierModel()
        et.n_estimators = 100
        et.random_state = self.seed
        et.search_models(x, y)

        ft = pd.DataFrame([x.columns.values, et.best_model.feature_importances_],
                          index=['Feature', 'Weight']).transpose().sort_values(by=['Weight'], ascending=False)
        limit = ft['Weight'][(ft['Feature'] == 'RANDOM_NUMBER')].iloc[0]
        feature_names = ft[ft['Weight'] > limit].sort_values(by=['Weight'], ascending=False)['Feature']
        self.best_features = list(feature_names)
        x_select = self.select_features_in_test_set(x)

        return x_select

