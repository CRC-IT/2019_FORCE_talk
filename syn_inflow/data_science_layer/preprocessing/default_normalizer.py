
from .abstract_pre_processor import AbstractPreProcessor
from sklearn.preprocessing import Normalizer


class DefaultNormalizer(AbstractPreProcessor):
    # l1(least absolute devations) - insensitive to outliers
    # l2(least squares) - takes outliers in consideration during training
    _normalizer = None
    normtype = 'l2'
    copy = True

    def fit_transform(self, data, y=None):
        self.fit(data, y)
        return self.transform(data, y)

    def transform(self, data, y=None):
        data = self._check_input(data)
        output = self._normalizer.transform(data)
        output = self._check_output(data, output)
        return output

    def fit(self, data, y=None):
        self._normalizer = Normalizer(norm=self.normtype, copy=self.copy)
        self._normalizer.fit(data)


