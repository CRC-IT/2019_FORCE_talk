
from .abstract_pre_processor import AbstractPreProcessor
from sklearn.preprocessing import RobustScaler


class RobustScalerTransformation(AbstractPreProcessor):

    _robust_scaler = None
    with_centering = True
    with_scaling = True
    quantile_range = (25.0, 75.0)
    copy = True

    def fit_transform(self, data, y=None):
        self.fit(data, y)
        return self.transform(data, y)

    def transform(self, data, y=None):
        data = self._check_input(data)
        output = self._robust_scaler.transform(data)
        output = self._check_output(data, output)
        return output

    def fit(self, data, y=None):
        self._robust_scaler = RobustScaler(with_centering=self.with_centering, with_scaling=self.with_scaling,
                                           quantile_range=self.quantile_range, copy=self.copy)
        self._robust_scaler.fit(data)
