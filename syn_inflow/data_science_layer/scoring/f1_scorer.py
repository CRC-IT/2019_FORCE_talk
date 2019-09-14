from .abstract_scorer import AbstractScorer
from sklearn.metrics import f1_score


class F1Scorer(AbstractScorer):
    greater_is_better = True
    needs_proba = False

    def score(self, true_y, predicted_y, **kwargs):
        score = f1_score(true_y, predicted_y, **kwargs)

        return score
