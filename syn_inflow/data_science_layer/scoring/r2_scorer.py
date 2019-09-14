from .abstract_scorer import AbstractScorer
from sklearn.metrics import r2_score


class R2Scorer(AbstractScorer):
    greater_is_better = True
    needs_proba = False

    def score(self, true_y, predicted_y, **kwargs):
        score = r2_score(true_y, predicted_y, **kwargs)

        return score
