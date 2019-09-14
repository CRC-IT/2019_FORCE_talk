
from ..data_split.abstract_test_train_split import\
    AbstractTestTrainSplit
from sklearn.model_selection import train_test_split


class DataFrameSequenceSplitter(AbstractTestTrainSplit):
    def train_test_split(
            self, *, table, y, train_size=None, random_state=None):
        x_indicies_list = list(range(table.shape[0]))
        train_idx, test_idx = train_test_split(
            x_indicies_list, train_size=train_size, random_state=random_state)
        return table[train_idx, :, :], table[test_idx, :, :],\
                     y[train_idx, :, :], y[test_idx, :, :]
