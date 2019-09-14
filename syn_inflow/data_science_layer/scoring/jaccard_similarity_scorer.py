from .abstract_scorer import AbstractScorer
from sklearn.metrics import jaccard_similarity_score


class JaccardSimilarityScorer(AbstractScorer):
    greater_is_better = True
    needs_proba = False

    def score(self, true_y, predicted_y, **kwargs):
        score = jaccard_similarity_score(true_y, predicted_y, **kwargs)

        return score
