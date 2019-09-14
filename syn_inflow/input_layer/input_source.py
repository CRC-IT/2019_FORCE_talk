# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 10:35:50 2017

@author: jonesn
"""
from abc import ABC, abstractmethod
from .utility.api_validator import ApiValidator
from .utility.column_cleaner import Column_Cleaner

class InputSource(ABC):
    """Abstract Factory Class Object defining methods and variables that must be provided by input sources"""
    table_name = ''
    name = ''
    filter_var = None
    filter = ''
    filter_date_var = None
    start_date = None
    end_date = None
    _ignore_var_filters = False
    _use_api_validator = True
    _set_api_length = 14

    @abstractmethod
    def return_data_dictionary(self):
        raise (NotImplementedError('Do not call base class method'))

    @staticmethod
    def _validate_api_filter_column(table, filter_var, use_api_validator=True, set_api_length=14):
        if use_api_validator and filter_var:
            table[filter_var] = ApiValidator().validate_api(
                api_list=list(table[filter_var]), api_length=set_api_length)
        else:
            pass
        return table

    @staticmethod
    def _clean_column_names(table, table_name):
        table.columns = Column_Cleaner().get_clean_columns(
            source_name=table_name,
            column_names=table.columns
            )
        return table

    @staticmethod
    def _get_filter_var_list(filter_var, api_list):
        # Set filter values
        if  filter_var == 'API_NO10':
            filter_var_list = [x[:10] for x in api_list]
        elif  filter_var == 'PID12':
            filter_var_list = [x[:12] for x in api_list]
        elif  filter_var == 'PID':
            filter_var_list = [x[:12] for x in api_list]
        elif  filter_var == 'api_no12':
            filter_var_list = [x[:12] for x in api_list]
        elif filter_var == 'api_no10':
            filter_var_list = [x[:10] for x in api_list]
        elif  filter_var == 'pid12':
            filter_var_list = [x[:12] for x in api_list]
        else:
            filter_var_list = api_list

        return filter_var_list




