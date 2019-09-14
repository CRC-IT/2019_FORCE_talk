from .abstract_scorer import AbstractScorer
from sklearn.metrics import balanced_accuracy_score


class BalancedAccuracyScorer(AbstractScorer):
    greater_is_better = True
    needs_proba = False

    def score(self, true_y, predicted_y, **kwargs):
        score = balanced_accuracy_score(true_y, predicted_y, **kwargs)

        return score
