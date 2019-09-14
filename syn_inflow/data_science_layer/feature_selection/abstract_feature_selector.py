
from abc import ABC, abstractmethod


class AbstractFeatureSelector(ABC):

    def __init__(self):
        self.best_features = None
        self.pre_fit_model = None
        self.selected_features_model = None
        self.seed = 1
        self.norm_order = 1
        self.threshold = 0.5
        self.prefit = False
        self.max_features = None

    @classmethod
    @abstractmethod
    def select_features(cls, x, y):
        raise NotImplementedError('Do not use ABC method')

    @abstractmethod
    def select_features_from_model(self, x, y):
        raise NotImplementedError('Do not use ABC method')

    def select_features_in_test_set(self, x):
        x_select = x[self.best_features]
        return x_select
