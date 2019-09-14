import pkg_resources
import pandas as pd
import warnings
from collections import defaultdict
from syn_inflow.data_science_layer.utilities.singleton import Singleton


class Column_Cleaner(object, metaclass=Singleton):
    """Class to clean up raw input source column names to the internal names used in the package

    ----- Public Methods -----
    get_clean_columns(self, *, source_name, column_names)
    get_clean_column(self, *, source_name, column_name)

    Clean Column names defined in packageData/database_name_cleaner_tables/columnNameCleanerDictionary.csv
    """
    def __init__(self):

        file_name = pkg_resources.resource_filename('syn_inflow', 'package_data/database_name_cleaner_tables/columnNameCleanerDictionary.csv')
        column_table = pd.read_csv(file_name)
        self.ColumnDictionary = defaultdict(list)
        self.dirtyTableDict = defaultdict(list)
        for idx, row in enumerate(column_table['DirtyColumn']):
            self.dirtyTableDict[column_table['DirtyTable'][idx]].append(column_table['DirtyColumn'][idx])
            if column_table['DirtyColumn'][idx] in self.ColumnDictionary.keys( ):
                tempDict = self.ColumnDictionary[column_table['DirtyColumn'][idx]]

                tempDict[column_table['DirtyTable'][idx]] = column_table['CleanColumn'][idx]

                self.ColumnDictionary[column_table['DirtyColumn'][idx]] = tempDict
            else:
                self.ColumnDictionary[column_table['DirtyColumn'][idx]] = {
                    column_table['DirtyTable'][idx]: column_table['CleanColumn'][idx]}

    def get_clean_columns(self, *, source_name, column_names):
        output = []
        for column in column_names:
            output.append(self.get_clean_column(source_name=source_name, column_name=column))
        return output

    def get_clean_column(self, *, source_name, column_name):
        try:
            lvl_one_dict = self.ColumnDictionary[column_name]
            try:
                cleanName = lvl_one_dict[source_name]
            except TypeError:
                warnings.warn(
                    'Warning! The column name provided has no clean name, please provide it to the data input name cleaner prevent future problems with external data dependancies.')
                return column_name
            if len(cleanName) < 1:
                warnings.warn(
                    'Warning! The column name provided has no clean name, please provide it to the data input name cleaner prevent future problems with external data dependancies.')
                return column_name
            return cleanName
        except KeyError:
            warnings.warn(
                'Warning! The column name provided has no clean name, please provide it to the data input name cleaner prevent future problems with external data dependancies.')
            return column_name
