
from .abstract_pre_processor import AbstractPreProcessor
from sklearn.impute import SimpleImputer


class ImputeMissing(AbstractPreProcessor):
    imputer = None
    strategy = 'mean'

    def fit(self, data, y=None):
        self.imputer = SimpleImputer(strategy=self.strategy)
        self.imputer.fit(data)

    def fit_transform(self, data, y=None):
        self.fit(data, y)
        return self.transform(data, y)

    def transform(self, data, y=None):
        data = self._check_input(data)
        output = self.imputer.transform(data)
        output = self._check_output(data, output)
        return output
