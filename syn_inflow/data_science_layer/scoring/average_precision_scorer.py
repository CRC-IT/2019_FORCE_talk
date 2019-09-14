from .abstract_scorer import AbstractScorer
from sklearn.metrics import average_precision_score


class AveragePrecisionScorer(AbstractScorer):
    greater_is_better = True
    needs_proba = True

    def score(self, true_y, predicted_y, **kwargs):
        probability_y = predicted_y

        score = average_precision_score(true_y, probability_y, **kwargs)

        return score
