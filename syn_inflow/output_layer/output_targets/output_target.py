
from abc import ABC, abstractmethod
import pandas as pd


class OutputTarget(ABC):
    """Abstract base class for output targets.
    Currently works with pandas tables.
    May be extended as needed to other output types"""
    format = 'csv'
    available_formats = {
        'csv': pd.DataFrame.to_csv,
        'pkl': pd.DataFrame.to_pickle,
        'json': pd.DataFrame.to_json
    }

    @classmethod
    def write_to_target(cls, data, name):
        raise (ValueError('Override this function'))

    def get_write_func(self):
        return self.available_formats[self.format]
