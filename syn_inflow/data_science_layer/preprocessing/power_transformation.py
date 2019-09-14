
from .abstract_pre_processor import AbstractPreProcessor
from sklearn.preprocessing import PowerTransformer


class PowerTransformation(AbstractPreProcessor):

    _power_transformer = None
    method = 'yeo - johnson'
    standardize = True
    copy = True

    def fit_transform(self, data, y=None):
        self.fit(data, y)
        return self.transform(data, y)

    def transform(self, data, y=None):
        data = self._check_input(data)
        output = self._power_transformer.transform(data)
        output = self._check_output(data, output)
        return output

    def fit(self, data, y=None):
        self._power_transformer = PowerTransformer(
            method = self.method, standardize = self.standardize, copy = self.copy)
        self._power_transformer.fit(data)