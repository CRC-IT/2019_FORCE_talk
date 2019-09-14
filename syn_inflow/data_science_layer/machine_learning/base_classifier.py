from .abstract_ml import AbstractML
from ..scoring.accuracy_scorer import AccuracyScorer


class BaseClassifier(AbstractML):

    def __init__(self):
        super().__init__()
        self.scorer = AccuracyScorer()
        self.search_scoring = AccuracyScorer()
        self.upper_threshold = None
        self.lower_threshold = None
        self.search_n_iter = 10
        self.search_verbose = 0
        self.search_random_state = 1

    def predict(self, x) -> (object, object):
        if self.best_model is None:
            raise (ValueError('Model is not fit! Fit model before trying to predict value(s)'))
        return self.current_model.predict(x), self.current_model.predict_proba(x)

    def _predict(self, x) -> (object, object):
        return self.current_model.predict(x), self.current_model.predict_proba(x)

    def set_params(self):
        super().set_params()
        self.sklearn_model.probability = True

