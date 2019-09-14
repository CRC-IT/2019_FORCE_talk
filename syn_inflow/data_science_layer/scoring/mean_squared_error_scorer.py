from .abstract_scorer import AbstractScorer
from sklearn.metrics import mean_squared_error


class MeanSquaredErrorScorer(AbstractScorer):
    greater_is_better = False
    needs_proba = False

    def score(self, true_y, predicted_y, **kwargs):
        score = mean_squared_error(true_y, predicted_y, **kwargs)

        return score
