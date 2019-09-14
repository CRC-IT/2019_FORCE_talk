from .abstract_scorer import AbstractScorer
from sklearn.metrics import precision_score


class PrecisionScorer(AbstractScorer):
    greater_is_better = True
    needs_proba = False

    def score(self, true_y, predicted_y, **kwargs):
        score = precision_score(true_y, predicted_y, **kwargs)

        return score
