from .abstract_scorer import AbstractScorer
from sklearn.metrics import fbeta_score


class FbetaScorer(AbstractScorer):
    greater_is_better = True
    needs_proba = False

    def score(self, true_y, predicted_y, **kwargs):
        score = fbeta_score(true_y, predicted_y, **kwargs)

        return score
