

from ..data_split.abstract_test_train_split import AbstractTestTrainSplit
from sklearn.model_selection import train_test_split
import pandas as pd


class SplitWithIndex(AbstractTestTrainSplit):
    index_lvl = 0

    # TODO: Add in training label Y input option
    def train_test_split(self, *, table, y, train_size, random_state=None):
        table, y = super( ).train_test_split(table=table, y=y)

        if isinstance(table.index, pd.MultiIndex):
            split_list = list(set(table.index.levels[self.index_lvl]))
        else:
            split_list = list(set(table.index))
        train, test = train_test_split(split_list, train_size=train_size, random_state=random_state)
        # TODO: handle lower level multi-indexing
        train_table = table.loc[train]
        train_y = y.loc[train]
        test_table = table.loc[test]
        test_y = y.loc[test]

        return train_table, test_table, train_y, test_y
