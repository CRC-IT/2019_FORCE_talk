
import os
from syn_inflow.data_science_layer.pipeline.transfer_learn_pipeline import *
import datetime as dttm
from syn_inflow.data_science_layer.synthetic_data_generator.addpay_generator import\
    AddpayGenerator
from syn_inflow.output_layer.output_targets.cache_output_target import CacheOutputTarget
from syn_inflow.data_layer.utilities.misc_functions import *
import pkg_resources


teams_dict = {
    'STEVENS': ['A3-6SND', 'MBB/W31', '31SNASH', 'N&T', '31SCDSH'],
             }

if __name__ == "__main__":
    skip_list = [
        # 'A3-6SND', #'31SNASH', 'N&T', '31SCDSH',
    ]
    run_no = '13'
    hold_out_date = dttm.datetime(2016, 1, 1)
    validation_funcs = [
        has_completion_height_and_production,
        log_production,
        create_log_inferred_prod,
        filter_very_low_or_absent_production,
        remove_infs,
        clean_up_dates,
        ]

    teams = list(teams_dict.keys())
    target_features = ['log_oil_rate', 'log_water_rate', 'log_gas_rate']

    for team in teams:
        package_data_folder = pkg_resources.resource_filename('syn_inflow', '/package_data/')
        x_cols = [
            'log_inferred_gas_from_nearest_producer',
            'log_inferred_oil_from_nearest_producer',
            'log_inferred_water_from_nearest_producer',
            'Completion_Height',
            'lift_type', 'Est_Remaining_Oil',  # 'GOR',
        ]
        reservoirs = teams_dict[team]

        if os.path.exists(package_data_folder + 'feature_data_{}.csv'.format(team)):
            actual_feature_data = pd.read_csv(
                package_data_folder + 'feature_data_{}.csv'.format(team))

        else:
            continue

        if os.path.exists(package_data_folder + 'inference_Data_{}.csv'.format(team)):
            inference_data = pd.read_csv(package_data_folder + 'inference_Data_{}.csv'.format(team))
        else:
            continue
        if 'inferred_oil_from_nearest_producer' not in inference_data:
            continue
        actual_feature_data_corrected_all = validate_table(
            actual_feature_data, validation_funcs)
        inference_data_corrected_all = validate_table(
            inference_data, validation_funcs)
        all_data_ = actual_feature_data_corrected_all.copy()

        for reservoir in reservoirs:
            try:
                inference_data_corrected = inference_data_corrected_all[
                    inference_data_corrected_all['RESERVOIR'] == reservoir]
            except KeyError:
                print('here')
            if reservoir in skip_list:
                continue
            print('starting team {} reservoir {}'.format(team, reservoir))
            if '/' in reservoir:
                temp_reservoir = reservoir.replace('/', '-')
            else:
                temp_reservoir = reservoir
            file = 'synthetic_{}_{}.csv'.format(team, temp_reservoir)
            actual_feature_data_corrected = actual_feature_data_corrected_all[
                actual_feature_data_corrected_all['RESERVOIR'] == reservoir]
            super_test_set = actual_feature_data_corrected[
                actual_feature_data_corrected['DATE'] > hold_out_date]
            if super_test_set.empty:
                super_test_set = actual_feature_data_corrected[
                    actual_feature_data_corrected['DATE'] > dttm.datetime(
                        2010,
                        1,
                        1)]
                actual_feature_data_corrected = \
                    actual_feature_data_corrected[
                        actual_feature_data_corrected[
                            'DATE'] <= dttm.datetime(
                            2010, 1, 1)]
            else:
                actual_feature_data_corrected = \
                actual_feature_data_corrected[
                    actual_feature_data_corrected['DATE'] <= hold_out_date]
            all_data = all_data_[
                all_data_['RESERVOIR'] == reservoir]
            super_test_set = super_test_set[
                super_test_set['RESERVOIR'] == reservoir]
            field = get_field_params_dict(team, reservoir)
            if not os.path.exists(file):
                if field is None:
                    continue
                generator = AddpayGenerator()
                synthetic_data = generator.generate_table(
                    200000, field=field, vars=actual_feature_data_corrected)
                synthetic_data.to_csv(file)
            else:
                synthetic_data = pd.read_csv(file)
            synthetic_data = validate_table(synthetic_data, validation_funcs)

            y_cols = [x for x in synthetic_data if 'True' in x]
            y_cols += ['lithology', 'viscosity', 'oil_rate', 'water_rate',
                       'free_gas_rate', 'solution_gas_rate']
            y_cols += ['oil_kh', 'water_kh', 'gas_kh', 'VRR', 'skin',
                       'Current_Recovery_Oil_4000', 'Current_Recovery_Oil_500']
            y_cols += ['perm_avg_ft', 'avg_porosity_ft', 'gross_rate',
                       'oil_per_ft',
                       'water_per_ft', 'gas_per_ft', 'log_oil_rate',
                       'log_gas_rate',
                       'log_water_rate', 'log_gross_rate']
            training_vars = synthetic_data[x_cols]
            target_vars = synthetic_data[y_cols]
            target_vars['gas_rate'] = target_vars['free_gas_rate'] + \
                                      target_vars['solution_gas_rate']
            pipeline = AddPayPipeline()
            model_path = pkg_resources.resource_filename('crcdal', '/cache/')
            model_name = 'model_{}_{}'.format(team, temp_reservoir)
            model_path += model_name
            pytorch_regressor = PytorchRegressor()
            if os.path.exists(model_path):
                pytorch_regressor.set_cached_model_path(model_name)
                pytorch_regressor.sklearn_model.fit_model = False
            else:
                pytorch_regressor.set_model_save_path(
                    model_name)
                pytorch_regressor.sklearn_model.load_cached_model = None
            pipeline._ml_models = [pytorch_regressor]
            pipeline._reports = []
            pipeline.fit(training_vars, target_vars[target_features])
            synthetic_data = add_predictions(
                synthetic_data, pipeline(training_vars))
            all_data = add_predictions(
                all_data, pipeline(all_data[x_cols]),
                columns=['syn_pred_oil', 'syn_pred_water', 'syn_pred_gas'])
            direct_pipeline = AddPayPipeline()
            direct_pipeline._reports = []
            direct_pipeline.define_default_models()
            direct_train = prep_table(
                actual_feature_data_corrected, x_cols, target_features)
            direct_pipeline.fit(direct_train[x_cols],
                                direct_train[target_features])
            all_data = prep_table(
                all_data, x_cols, target_features)
            if not super_test_set.empty:
                super_test_set = prep_table(
                    super_test_set, x_cols, target_features)
            actual_feature_data_corrected = prep_table(
                actual_feature_data_corrected, x_cols, target_features)
            all_data = add_predictions(
                all_data, pipeline(all_data[x_cols]),
                columns=['all_pred_oil', 'all_pred_water', 'all_pred_gas'])
            if not super_test_set.empty:
                super_test_set = add_predictions(
                    super_test_set, pipeline(super_test_set[x_cols]))
            actual_feature_data_corrected = add_predictions(
                actual_feature_data_corrected, pipeline(
                    actual_feature_data_corrected[x_cols]),
                columns=['syn_pred_oil', 'syn_pred_water', 'syn_pred_gas'])
            if not super_test_set.empty:
                super_test_set = add_predictions(
                    super_test_set, direct_pipeline(super_test_set[x_cols]),
            columns=['direct_pred_oil', 'direct_pred_water', 'direct_pred_gas'])
            all_data = add_predictions(
                all_data, direct_pipeline(all_data[x_cols]),
                columns=['direct_pred_oil', 'direct_pred_water',
                         'direct_pred_gas'])
            all_data['Team'] = team
            all_data['RESERVOIR'] = reservoir
            all_data['RunNum'] = run_no
            linear_models = linear_model_fit(
                pipeline(actual_feature_data_corrected[x_cols]),
                actual_feature_data_corrected[target_features],
                target_features
            )
            all_data = add_predictions(all_data, apply_linear_models(
                pipeline(all_data[x_cols]),
                linear_models, target_features
            ),
            columns=['cond_pred_oil',
                    'cond_pred_water',
                    'cond_pred_gas']
            )
            try:
                super_test_set = add_predictions(
                    super_test_set,
                    apply_linear_models(
                        pipeline(super_test_set[x_cols]),
                        linear_models, target_features
                    ),
                    columns = ['cond_pred_oil',
                          'cond_pred_water',
                          'cond_pred_gas']
                )
            except Exception as ex:
                print(ex)
            inference_data_corrected = prep_table(
                inference_data_corrected, x_cols,
                target_features, inference_table=True)

            if not inference_data_corrected.empty:
                inference_data_corrected = add_predictions(
                    inference_data_corrected,
                    pipeline(inference_data_corrected[x_cols]),
                    columns=['all_pred_oil', 'all_pred_water', 'all_pred_gas']
                )
                inference_data_corrected = add_predictions(
                    inference_data_corrected,
                    direct_pipeline(inference_data_corrected[x_cols]),
                    columns=['direct_pred_oil',
                             'direct_pred_water',
                             'direct_pred_gas']
                )
                inference_data_corrected = add_predictions(
                    inference_data_corrected,
                    apply_linear_models(
                        pipeline(inference_data_corrected[x_cols]),
                        linear_models, target_features
                    ),
                    columns=['cond_pred_oil',
                            'cond_pred_water',
                            'cond_pred_gas']
                )
                inference_data_corrected['Team'] = team
                inference_data_corrected['RESERVOIR'] = reservoir
                inference_data_corrected['RunNum'] = run_no
                CacheOutputTarget.write_to_target(inference_data_corrected,
                                                'ml_results_inference_data')
            CacheOutputTarget.write_to_target(all_data, 'ml_results_all_data')
            if not super_test_set.empty:
                super_test_set['Team'] = team
                super_test_set['RESERVOIR'] = reservoir
                super_test_set['RunNum'] = run_no
                CacheOutputTarget.write_to_target(
                    super_test_set, 'ml_results_verification_data')
            synthetic_data['Team'] = team
            synthetic_data['RESERVOIR'] = reservoir
            synthetic_data['RunNum'] = run_no
            CacheOutputTarget.write_to_target(
                synthetic_data, 'ml_results_synthetic_data')
            print('finished: {}'.format(reservoir))
