from abc import ABC, abstractmethod


class AbstractScorer(ABC):
    greater_is_better = True
    needs_proba = False

    @abstractmethod
    def score(self, true_y, predicted_y, **kwargs):
        raise (NotImplementedError('Do not use abstract class method'))
