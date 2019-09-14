'''
Authored by Nathaniel Jones,
Modified and maintained by the Big Data Analytics Team at California Resources Corporation
CC BY 2.0 License
'''

from .basic_pipeline import BasicPipeline
from crcdal.data_layer.data_interfaces.add_pay_synthetic_data_interface \
    import \
    AddPaySyntheticDataInterface
from .add_pay_pipeline import AddPayPipeline
from ..machine_learning.pytorch_nn_regressor import PytorchRegressor
import numpy as np
from crcdal.data_layer.data_interfaces.past_completions_interface import \
    PastCompletionsInterface
import pandas as pd
import datetime as dttm
from crcdal.output_layer.utilities.aws_s3_funcs import *
from crcdal.input_layer.configuration import Configuration
from crcdal.data_layer.data_interfaces.add_pay_feature_formatting_interface\
    import AddPayFeatureFormattingInterface


class TransferLearnPipeline(BasicPipeline):
    training_cols = ['log_inferred_gas_from_nearest_producer',
                     'log_inferred_oil_from_nearest_producer',
                     'log_inferred_water_from_nearest_producer',
                     'Est_4000ft_Recovery', 'Completion_Height',
                     'lift_type', ]
    y_cols = ['log_oil_rate', 'log_water_rate', 'log_gas_rate']
    current_version = "0.0.1"

    def __init__(self, transfer_learn_pipeline=None, final_pipeline=None):
        super().__init__()
        self.transfer_pipe = transfer_learn_pipeline
        self.final_pipe = final_pipeline
        self.validation_funcs = [
            has_completion_height_and_production,
            create_log_inferred_prod,
            filter_very_low_or_absent_production,
            remove_infs,
            clean_up_dates,
        ]
        self.bucket_name = 'transfer-pipe-' + \
                           Configuration().get_cache_subfolder()
        self.bucket_id = get_bucket_id_with_string(self.bucket_name)
        self.holdout_data = pd.DataFrame()

    def __call__(self, data, y_result=None):
        output = self.transfer_pipe(data)
        return self.final_pipe(output)

    @classmethod
    def train_pipe_for_team(cls, api_list, team):
        # pull in data
        synthetic_data_interface = AddPaySyntheticDataInterface. \
            get_interface_with_name(name=team, api_list=api_list)
        synthetic_data = synthetic_data_interface.get_synthetic_data(
            size=100000)
        obj = cls(cls.create_transfer_pipe(), cls.create_final_pipe())
        synthetic_data = obj.validate_table(synthetic_data)
        obj.fit_transfer_pipe(
            synthetic_data[cls.training_cols],
            synthetic_data[cls.y_cols])
        interface = PastCompletionsInterface.get_interface_with_name()
        actual_data = interface.make_transfer_learning_feature_table()
        actual_data = obj.validate_table(actual_data)
        super_test_set, actual_data = obj.create_super_test_set_after_date(
            dttm.datetime(2009, 1, 1), actual_data
        )
        obj.fit(actual_data[
                    obj.training_cols],
                actual_data[obj.y_cols])
        obj.holdout_data = super_test_set
        obj._report_results()
        # cache deployment pipeline
        write_pickle(
            obj.transfer_pipe,
            team + obj.current_version + '-transfer_pipe.pkl', obj.bucket_id)
        write_pickle(
            obj.final_pipe, team + obj.current_version + '-final_pipe.pkl',
            obj.bucket_id)
        return obj

    def search_pipeline_parameter_space(self):
        # TODO Search feature combos, model types, model hyper-parameters
        pass

    def retrieve_model_params(self, api_list, team):
        # get cached model
        self._prep_cached_model(team)

        # make a report table and return it
        pass

    def deploy_pipe_for_team(self, api_list, team):
        # get cached model
        self._prep_cached_model(team)
        interface = AddPayFeatureFormattingInterface.get_interface_with_name()
        data = interface.make_rate_prediction_table()
        data = self.validate_table(data)
        # run predict job
        results = self(data)
        #TODO Deployment Reports

        # return results
        return results

    def validate_table(self, table):
        for func in self.validation_funcs:
            table = func(table)
        return table

    def _prep_cached_model(self, team):
        if self.transfer_pipe is None or self.final_pipe is None:
            self.final_pipe = load_pickle_from_s3(
                team + self.bucket_id, self.current_version +
                '-final_pipe.pkl')
            self.transfer_pipe = load_pickle_from_s3(
                team + self.bucket_id, self.current_version +
                '-transfer_pipe.pkl')

    @staticmethod
    def create_transfer_pipe(model_path=None):
        pipeline = AddPayPipeline()
        pytorch_regressor = PytorchRegressor()
        if model_path is not None:
            pytorch_regressor.set_cached_model_path(model_path)
        pipeline._ml_models = [pytorch_regressor]
        pipeline._reports = []
        return pipeline

    @staticmethod
    def create_final_pipe():
        pipeline = AddPayPipeline()
        pipeline.train_fraction = 0.8
        pipeline._reports = []
        return pipeline

    def fit_transfer_pipe(self, data, training_y):
        self.transfer_pipe.fit(data, training_y)

    def fit(self, data, training_y):
        output = self.transfer_pipe(data)
        self.final_pipe.fit(output, training_y)
        # TODO make reports
        print('reports here')
        # TODO train using all data
        pass

    def create_super_test_set_after_date(self, date, table):
        super_test_set = table[
            table['DATE'] > date]
        training_data = table[table['DATE'] <= date]
        return super_test_set, training_data


def log_production(table, production_streams=('oil', 'water', 'gas')):
    if 'log_oil_rate' not in table:
        for stream in production_streams:
            if stream in table:
                table['log_{}_rate'.format(stream)] = \
                    np.log10(table[stream])
                table['log_{}_rate'.format(stream)].replace(
                    [np.inf, -np.inf, None], -3.0
                )
            else:
                table['log_{}_rate'.format(stream)] = None
    return table

def create_log_inferred_prod(table):
    import numpy as np
    production_streams = ['oil', 'water', 'gas']
    for stream in production_streams:
        table['log_inferred_' + stream + '_from_nearest_producer'] = \
            np.log10(table['inferred_' + stream +
                           '_from_nearest_producer'])
        table['log_inferred_' + stream + '_from_nearest_producer'].replace(
            [np.inf, -np.inf, None], -3.0
        )
    return table


def has_completion_height_and_production(table):
    table = table[table['Completion_Height'] > 1]
    if 'gross_prod' in table:
        table = table[table['gross_prod'] > 0]
    return table


def remove_infs(table):
    table = table.replace([np.inf, -np.inf, None], np.nan)
    return table


def filter_very_low_or_absent_production(table):
    if 'log_oil_rate' in table:
        oil_filter = table['log_oil_rate'] < -2.9
        gas_filter = table['log_gas_rate'] < -2.9
        water_filter = table['log_water_rate'] < -2.9
        combined = oil_filter & gas_filter & water_filter
        table = table[~combined]
    return table


def clean_up_dates(table):
    if 'DATE' in table:
        table['DATE'] = pd.to_datetime(
            table['DATE'])
    return table
