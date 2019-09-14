from .abstract_scorer import AbstractScorer
from sklearn.metrics import accuracy_score


class AccuracyScorer(AbstractScorer):
    greater_is_better = True
    needs_proba = False

    def score(self, true_y, predicted_y, **kwargs):
        score = accuracy_score(true_y, predicted_y, **kwargs)

        return score
