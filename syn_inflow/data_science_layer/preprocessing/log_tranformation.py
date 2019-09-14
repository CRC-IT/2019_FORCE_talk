
from .abstract_pre_processor import AbstractPreProcessor
import pandas as pd
import numpy as np


class LogTransformation(AbstractPreProcessor):

    def fit_transform(self, data, y=None):

        return self._process_data(data=data)

    def fit(self, data, y=None):
        return None

    def transform(self, data, y=None):
        return self._process_data(data=data)

    def _process_data(self, data):
        try:
            output = np.log10(data)
        except Exception as e:
            print(e)
            output = np.log10(np.float32(data.values))
            output = self._check_output(data, output)
        return output

    def _check_output(self, input, output):
        if isinstance(input, pd.DataFrame):
            output = pd.DataFrame(data=output, index=input.index, columns=input.columns)
        return output
