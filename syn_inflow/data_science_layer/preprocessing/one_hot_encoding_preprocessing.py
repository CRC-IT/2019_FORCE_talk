import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from .abstract_pre_processor import AbstractPreProcessor


class OneHotProcessing(AbstractPreProcessor):
    _label_encoders = None
    _one_hot_encoders = None

    def __init__(self):
        _label_encoders = None
        _one_hot_encoders = None

    def fit_transform(self, data, y=None) -> (object, object, object):
        self._fit(data)
        self._check_is_fit()
        return self.transform(data)

    def transform(self, data, y=None) -> (object, object):
        self._check_is_fit()
        return self._transform(data)

    def fit(self, data, y=None) -> (object, object):
        self._fit(data)

    def _transform(self, data):
        data = data.reset_index()
        int_cols, string_cols = self._get_types(data)
        for idx, col in enumerate(string_cols):
            data[data.columns[col] + 'encoded'] = self._label_encoders[idx].transform(data.iloc[:, col])
            int_cols.append(data.shape[1] - 1)
        for idx, col in enumerate(int_cols):
            data_col = data.iloc[:, col].values
            data_col = data_col.reshape(-1, 1)
            transformed_array = self._one_hot_encoders[idx].transform(data_col)
            transformed_array = transformed_array.toarray()
            dataframe_temp = pd.DataFrame(transformed_array)
            for idx, col2 in enumerate(dataframe_temp):
                data.loc[:, data.columns[col] + "_" + str(idx)] = dataframe_temp[col2]
        data = data.drop(list(data.columns[int_cols]) + list(data.columns[string_cols]), axis=1)
        data.set_index(keys=[data.columns[0]], inplace=True)
        return data

    def _fit(self, data):
        int_cols, string_cols = self._get_types(data)
        if self._label_encoders is None:
            self._label_encoders = []
            self._one_hot_encoders = []
        for col in string_cols:
            encoder = LabelEncoder()
            encoder.fit(y=data.iloc[:, col])
            transformedCol = encoder.transform(data.iloc[:, col])
            data[data.columns[col] + 'encoded'] = transformedCol
            #data = data.drop(data.columns[col], axis=1)
            int_cols.append(data.shape[1] - 1)
            self._label_encoders.append(encoder)
        for col in int_cols:
            encoder = OneHotEncoder()
            data_col = data.iloc[:, col].values
            data_col = data_col.reshape(-1, 1)
            encoder.fit(data_col)
            self._one_hot_encoders.append(encoder)

    def _check_is_fit(self):
        if self._label_encoders is None:
            raise IndexError('Encoders not fit to data')
        if self._one_hot_encoders is None:
            raise IndexError('Encoders not fit to data')

    @staticmethod
    def _get_types(data):
        string_cols = []
        int_cols = []
        for idx, col in enumerate(data):
            sample_val = data.iat[0, idx]
            if isinstance(sample_val, str):
                string_cols.append(idx)
            if isinstance(sample_val, int):
                int_cols.append(idx)
        return int_cols, string_cols