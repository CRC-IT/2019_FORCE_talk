
from .abstract_pre_processor import AbstractPreProcessor
import numpy as np
from sklearn.preprocessing import OneHotEncoder


class OneHotEncoderTransformation(AbstractPreProcessor):

    _one_hot_encoder = None
    n_values = 'auto'
    categorical_features = 'all'
    dtype = np.float
    sparse = True
    handle_unknown = 'error'

    def fit_transform(self, data, y=None):
        self.fit(data, y)
        return self.transform(data, y)

    def transform(self, data, y=None):
        data = self._check_input(data)
        output = self._one_hot_encoder.transform(data)
        output = self._check_output(data, output)
        return output

    def fit(self, data, y=None):
        self._one_hot_encoder = OneHotEncoder(n_values=self.n_values, categorical_features=self.categorical_features
                                            , dtype=self.dtype, sparse=self.sparse, handle_unknown=self.handle_unknown)
        self._one_hot_encoder.fit(data)
