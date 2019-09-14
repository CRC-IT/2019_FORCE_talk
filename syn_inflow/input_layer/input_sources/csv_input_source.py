'''
Authored by Nathaniel Jones, 
Modified and maintained by the Big Data Analytics Team
Copyright California Resources Corporation 2018, all rights reserved
'''

from ..input_source import InputSource
import pandas as pd
from collections import defaultdict


class CSVInputSource(InputSource):
    filename = None

    def __init__(self, filename, clean_name, filter_var=None, filter_date_var=None, table_name=''):
        self.filename = filename
        self.table_name = table_name
        self.filter_var = filter_var
        self.filter_date_var = filter_date_var
        self.name = clean_name

    def return_data_dictionary(self):
        output = defaultdict(list)
        data = pd.read_csv(self.filename, index_col=False)
        data = self._validate_api_filter_column(data, self.filter_var, self._use_api_validator, self._set_api_length)
        if self.filter and not self._ignore_var_filters:
            data = self._apply_var_filter(data)
        data = self._apply_date_filter(data)
        if data.shape[0] < 1:
            return defaultdict(list)
        data = self._clean_column_names(data, self.table_name)
        output[self.name] = data
        return output

    def _apply_var_filter(self, table) -> pd.DataFrame:
        try:
            if not isinstance(self.filter, list):
                self.filter = [self.filter]
            filter_var_list = self._get_filter_var_list(self.filter_var, self.filter)
            table = table[table[self.filter_var].isin(filter_var_list)]
        except Exception:
            print('filter failed')
        finally:
            return table

    def _apply_date_filter(self, table) -> pd.DataFrame:
        try:
            if self.start_date:
                table[self.filter_date_var] = pd.to_datetime(table[self.filter_date_var])
                table = table[table[self.filter_date_var] >= self.start_date]
            if self.end_date:
                table[self.filter_date_var] = pd.to_datetime(table[self.filter_date_var])
                table = table[table[self.filter_date_var] <= self.end_date]
        except Exception:
            print('date filter failed')
        finally:
            return table

