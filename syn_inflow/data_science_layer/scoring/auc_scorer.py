from .abstract_scorer import AbstractScorer
from sklearn.metrics import roc_auc_score


class AUCScorer(AbstractScorer):
    greater_is_better = True
    needs_proba = True

    def score(self, true_y, predicted_y, **kwargs):
        probability_y = predicted_y

        score = roc_auc_score(true_y, probability_y, **kwargs)

        return score
