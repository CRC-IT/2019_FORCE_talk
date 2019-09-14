from .abstract_scorer import AbstractScorer
from sklearn.metrics import mean_squared_error
import math


class RootMeanSquaredErrorScorer(AbstractScorer):
    greater_is_better = False
    needs_proba = False

    def score(self, true_y, predicted_y, **kwargs):
        score = math.sqrt(mean_squared_error(true_y, predicted_y, **kwargs))

        return score
