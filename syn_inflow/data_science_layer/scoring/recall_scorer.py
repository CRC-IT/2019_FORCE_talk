from .abstract_scorer import AbstractScorer
from sklearn.metrics import recall_score


class RecallScorer(AbstractScorer):
    greater_is_better = True
    needs_proba = False

    def score(self, true_y, predicted_y, **kwargs):
        score = recall_score(true_y, predicted_y, **kwargs)

        return score
