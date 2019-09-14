from .abstract_scorer import AbstractScorer
from sklearn.metrics import cohen_kappa_score


class CohenKappaScorer(AbstractScorer):
    greater_is_better = True
    needs_proba = False

    def score(self, true_y, predicted_y, **kwargs):
        score = cohen_kappa_score(true_y, predicted_y, **kwargs)

        return score
