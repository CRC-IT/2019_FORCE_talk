
from .abstract_pre_processor import AbstractPreProcessor
from sklearn.preprocessing import QuantileTransformer


class QuantileTransformation(AbstractPreProcessor):

    _quantile_transformer = None
    n_quantiles = 1000
    output_distribution = 'uniform'
    ignore_implicit_zeros = False
    subsample = 100000
    random_state = None
    copy = True

    def fit_transform(self, data, y=None):
        self.fit(data, y)
        return self.transform(data, y)

    def transform(self, data, y=None):
        data = self._check_input(data)
        output = self._quantile_transformer.transform(data)
        output = self._check_output(data, output)
        return output

    def fit(self, data, y=None):
        self._quantile_transformer = QuantileTransformer(
            n_quantiles=self.n_quantiles, output_distribution=self.output_distribution,
            ignore_implicit_zeros=self.ignore_implicit_zeros,
            subsample=self.subsample, random_state=self.random_state, copy=self.copy)
        self._quantile_transformer.fit(data)