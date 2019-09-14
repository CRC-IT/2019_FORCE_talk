from .abstract_ml import AbstractML
from ..scoring.r2_scorer import R2Scorer


class BaseRegressor(AbstractML):

    def __init__(self):
        self.search_n_iter = 10
        self.search_verbose = 0
        self.search_random_state = 1
        self.scorer = R2Scorer()
        self.search_scoring = R2Scorer()

    def predict(self, x) -> (object, object):
        if self.best_model is None:
            raise (ValueError('Model is not fit! Fit model before trying to predict value(s)'))
        return self.best_model.predict(x), None

    def _predict(self, x) -> (object, object):
        return self.current_model.predict(x), None
