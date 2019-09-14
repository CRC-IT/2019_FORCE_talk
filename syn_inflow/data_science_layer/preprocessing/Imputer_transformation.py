
from .abstract_pre_processor import AbstractPreProcessor
from sklearn.preprocessing import Imputer


class ImputerTransformation(AbstractPreProcessor):

    _imputer = None
    missing_values = 'NaN'
    strategy = 'mean'
    axis = 0
    verbose = 0
    copy = True

    def fit_transform(self, data, y=None):
        self.fit(data, y)
        return self.transform(data, y)

    def transform(self, data, y=None):
        data = self._check_input(data)
        output = self._imputer.transform(data)
        output = self._check_output(data, output)
        return output

    def fit(self, data, y=None):
        self._imputer = Imputer(
            missing_values=self.missing_values, strategy=self.strategy, axis=self.axis,
            verbose=self.verbose, copy=self.copy)
        self._imputer.fit(data)
