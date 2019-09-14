
from ..data_split.abstract_test_train_split import AbstractTestTrainSplit
from sklearn.model_selection import train_test_split


class TableSplit(AbstractTestTrainSplit):
    # TODO: Add in training label Y option
    def train_test_split(self, *, table, y, train_size, random_state=None):
        table, y = super( ).train_test_split(table=table, y=y)
        train, test = train_test_split(table, train_size=train_size, random_state=random_state)
        train_filt = list(set(train.index))
        test_filt = list(set(test.index))
        return train, test, y.loc[train_filt], y.loc[test_filt]
