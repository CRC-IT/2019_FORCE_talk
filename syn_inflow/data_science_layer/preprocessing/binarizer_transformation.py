
from .abstract_pre_processor import AbstractPreProcessor
from sklearn.preprocessing import Binarizer


class BinarizerTransformation(AbstractPreProcessor):

    _binarizer = None
    threshold = 0.0
    copy = True

    def fit_transform(self, data, y=None):
        self.fit(data, y)
        return self.transform(data, y)

    def transform(self, data, y=None):
        data = self._check_input(data)
        output = self._binarizer.transform(data)
        output = self._check_output(data, output)
        return output

    def fit(self, data, y=None):
        self._binarizer = Binarizer(
            threshold=self.threshold, copy=self.copy)
        self._binarizer.fit(data)