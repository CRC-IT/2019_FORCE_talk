
from .abstract_pre_processor import AbstractPreProcessor
import pandas as pd
from sklearn.preprocessing import StandardScaler


class DefaultScaler(AbstractPreProcessor):
    """Center and scale data based on filters defined for columns/data types. """
    _scaler = None

    copy = True
    with_mean = True
    with_std = True

    # TODO test this class using filters

    def fit_transform(self, data, y=None):
        self._fit_data(data, y)
        return self.transform(data, y)

    def transform(self, data, y=None):
        data = self._check_input(data)
        float_data_cols, non_float = self._get_types(data)
        float_data = data[float_data_cols]
        self._input_data = float_data
        output = self._transform_data(data)
        output = self._check_output(data, output)

        return output

    def fit(self, data, y=None):
        self._fit_data(data, y)

    def inverse_transform(self, data):
        return self._scaler.inverse_transform(data)

    def _transform(self, data, y):
        output = self._transform_data(data)
        return output

    def _fit_data(self, data, y):
        data = self._check_input(data)
        float_data_cols, non_float = self._get_types(data)
        float_data = data[float_data_cols]
        # temp_data = self._apply_filters(float_data)
        self._check_scaler( )
        self._scaler.fit(float_data, y)

    def _transform_data(self, data):
        self._check_scaler()
        data = self._check_input(data)
        float_data_cols, non_float = self._get_types(data)
        float_data = data[float_data_cols]
        # float_data = self._apply_filters(float_data)
        output = self._scaler.transform(float_data)

        output = self._check_output(float_data, output)
        output = pd.concat([output, data[non_float]], axis=1)
        return output

    def _check_scaler(self):
        if self._scaler is None:
            self._scaler = StandardScaler(copy=self.copy, with_mean=self.with_mean, with_std=self.with_std)

    def _get_types(self, data):
        float_cols = []
        non_float = []
        for idx, col in enumerate(data):
            sample_val = data.iat[0, idx]
            if isinstance(sample_val, str):
                non_float.append(col)
            elif isinstance(sample_val, int):
                non_float.append(col)
            else:
                float_cols.append(col)
        return float_cols, non_float
