
from .abstract_pre_processor import AbstractPreProcessor
from sklearn.preprocessing import MinMaxScaler

class MinMaxScalerTransformation(AbstractPreProcessor):

    feature_range = (0,1)
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
        self._power_transformer = MinMaxScaler(feature_range=self.feature_range, copy=self.copy)
        self._power_transformer.fit(data)