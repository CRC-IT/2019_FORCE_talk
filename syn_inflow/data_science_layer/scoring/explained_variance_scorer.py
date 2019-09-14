from .abstract_scorer import AbstractScorer
from sklearn.metrics import explained_variance_score


class ExplainedVarianceScorer(AbstractScorer):
    greater_is_better = True
    needs_proba = False

    def score(self, true_y, predicted_y, **kwargs):
        score = explained_variance_score(true_y, predicted_y, **kwargs)

        return score
