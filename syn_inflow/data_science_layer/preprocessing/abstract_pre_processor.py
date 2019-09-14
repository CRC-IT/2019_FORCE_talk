
from abc import ABC, abstractmethod
import pandas as pd


class AbstractPreProcessor(ABC):
    column_filter = None
    type_filter = None
    functional_filter = None
    filtered_out = None
    _input_data = None


    @abstractmethod
    def fit_transform(self, data, y=None) -> (object, object, object):
        raise (NotImplementedError('Do not use abc function'))

    @abstractmethod
    def transform(self, data, y=None) -> (object, object):
        raise (NotImplementedError('Do not use abc function'))

    def inverse_transform(self, data):
        pass

    @abstractmethod
    def fit(self, data, y=None) -> (object, object):
        raise (NotImplementedError('Do not use abc function'))

    def _check_output(self, data, output):
        if isinstance(data, pd.DataFrame):
            output = pd.DataFrame(data=output, columns=data.columns,index=data.index)
        return output

    def _check_input(self, data):
        if isinstance(data, pd.DataFrame):
            return data
        return pd.DataFrame(data)

    # def _apply_filters(self, data):
    #     original_cols = data.columns
    #     if self.column_filter is not None:
    #         data = data[self.column_filter]
    #     if self.type_filter is not None:
    #         # TODO: setup type filter
    #         pass
    #     if self.functional_filter is not None:
    #         data = self.functional_filter(data)
    #     self.filtered_out = [x for x in data.columns if x not in original_cols]
    #     if len(self.filtered_out) == 0:
    #         self.filtered_out = None
    #     return data
    #
    # def _recombine_filtered_and_unfiltered_cols(self, processed_data, original_data):
    #     if self.filtered_out is not None:
    #         return pd.concat([processed_data, original_data[self.filtered_out]])
    #     else:
    #         return processed_data
