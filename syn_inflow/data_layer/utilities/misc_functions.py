from syn_inflow.input_layer.input_sources.csv_input_source import CSVInputSource
import pkg_resources
from sklearn.metrics import mean_absolute_error
import numpy as np
import pandas as pd

def get_field_params_dict(team, reservoir):
    df = get_field_params(team, reservoir)
    if df.empty:
        return dict()
    series = df.iloc[0]
    return series.to_dict()

def get_field_params(team, reservoir):
    data = get_data_from_source(team)
    return data[data['RESERVOIR'] == reservoir]

def get_data_from_source(team):
    file = pkg_resources.resource_filename(
        'syn_inflow', '/package_data/Field_Generator_params.csv')
    source = CSVInputSource(
        filename=file,
        clean_name='field_params',
        filter_var='Data_Tag',
        table_name='field_params')
    source.filter = team
    data = source.return_data_dictionary()
    return data['field_params']

def linear_model_fit(predictions, real_data, target_features):
    output_models = []
    for idx, feature in enumerate(target_features):
        output_models.append(search_linear_params(
            pd.DataFrame(predictions[:, idx]), real_data.iloc[:, idx]))
    return output_models


def apply_linear_models(predictions, models, target_features):
    output = np.zeros(predictions.shape)
    for idx, feature in enumerate(target_features):
        output[:, idx] = predictions[:, idx] * models[idx][1] + models[idx][0]
    return output


def validate_table(table, funcs):
    for func in funcs:
        table = func(table)
    return table


def prep_table(table, x_cols, target_features, inference_table=False):
    table['targets_imputed'] = False
    table['imputed_vars'] = ''
    if not inference_table:
        afilter = table[
           target_features].isna().sum(axis=1)
        afilter = afilter == 3
        table = table.loc[~afilter]
        afilter = table[target_features].isna()

        for x in range(len(target_features)):
            table.loc[afilter.iloc[:, x], target_features[x]] = -2.0
            table.loc[
                afilter.iloc[:, x] | table['targets_imputed'],
                'targets_imputed'] = True
            table.loc[
                afilter.iloc[:, x], 'imputed_vars'] += target_features[x] + '_'

    afilter = table[x_cols].isna()
    table['inputs_imputed'] = False
    for x in range(len(x_cols)):
        mean_val = table[x_cols[x]].mean()
        if np.isnan(mean_val):
            mean_val = 0.0
        table.loc[afilter.iloc[:, x], x_cols[x]] = mean_val
        table.loc[
            afilter.iloc[:, x] | table['inputs_imputed'],
            'inputs_imputed'] = True
        table.loc[
            afilter.iloc[:, x], 'imputed_vars'] += x_cols[x] + '_'
    return table

def add_predictions(
        table, predictions, columns=['pred_oil', 'pred_water', 'pred_gas']):
    for idx, col in enumerate(columns):
        try:
            table[col] = predictions[:, idx]
        except ValueError as ex:
            print(ex)
    return table

def search_linear_params(predictions, real_data):
    y_params = np.arange(-2.0, 1, 0.05)
    m_params = np.arange(0.3, 1.6, 0.01)
    best_mae = 400
    best_y = 0
    best_m = 1
    for y in y_params:
        for m in m_params:
            adj_predictions = predictions * m + y
            current_mae = mean_absolute_error(
                real_data, adj_predictions)
            if current_mae < best_mae:
                best_mae = current_mae
                best_m = m
                best_y = y
    return best_y, best_m

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
