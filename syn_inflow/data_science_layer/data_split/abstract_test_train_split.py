
from abc import ABC, abstractmethod
import pandas as pd


class AbstractTestTrainSplit(ABC):
    @abstractmethod
    def train_test_split(self, *, table, y, train_size=None, random_state=None):
        return self._check_input(table, y)

    @staticmethod
    def _check_input(table, y):
        if not isinstance(table, pd.DataFrame):
            table = pd.DataFrame(table)
        if not isinstance(y, pd.DataFrame):
            if not isinstance(y, pd.Series):
                if len(y.shape) > 1:
                    if y.shape[1] > 0:
                        y = pd.DataFrame(y)
                else:
                    y = pd.Series(y)
        return table, y
