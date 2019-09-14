
from .abstract_pre_processor import AbstractPreProcessor
from sklearn.preprocessing import PolynomialFeatures


class PolynomialFeaturesTransformation(AbstractPreProcessor):

    _polynomial_features = None
    degree = 2
    interaction_only = False
    include_bias = True

    def fit_transform(self, data, y=None):
        self.fit(data, y)
        return self.transform(data, y)

    def transform(self, data, y=None):
        data = self._check_input(data)
        output = self._polynomial_features.transform(data)
        output = self._check_output(data, output)
        return output

    def fit(self, data, y=None):
        self._polynomial_features = PolynomialFeatures(
            degree=self.degree, interaction_only=self.interaction_only, include_bias=self.include_bias)
        self._polynomial_features.fit(data)
