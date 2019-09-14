from .data_generator import \
    DataGenerator, random
import math
from collections import defaultdict
import pandas as pd
from .addpay_field_generator import AddPayFieldGenerator
import numpy as np


class AddpayGenerator(DataGenerator):
    fields = None
    no_of_fields = 1

    def __init__(self):
        super().__init__()
        self.output = defaultdict()
        self.fields = {}
        self.current_field = None
        self.visocity_volume_fac = None
        self.visocity_volume_fac_gas = None
        self.injector_rate = None
        self.field_gen = AddPayFieldGenerator().return_synthetic_data()
        self.table_dict = defaultdict(list)
        self.p10 = 0
        self.p25 = 0
        self.p50 = 0
        self.p75 = 0
        self.p90 = 0
        self.p1 = 0
        self.well_count = 1000
        self.var_stats = {}

    @classmethod
    def generate_table_for_pipeline(cls, well_count=1000, seed=None,
                                    no_fields=1):
        if isinstance(seed, str):
            seed = int(seed)
        cls.fields = no_fields
        field_data_generator = cls()
        return field_data_generator.generate_table(well_count=well_count,
                                                   seed=seed)

    def generate_table(self, well_count, seed=None, field=None, vars=None):
        if vars is not None and field is not None:
            self.current_field = field
            self._handle_vars(vars)
        func = self.return_synthetic_data(field=field)
        counter = 0
        self.check_seed(seed)
        self.well_count = well_count
        for x in func:
            added = self._handle_output(x)
            if added:
                counter += 1
                self._progress_tracker(counter, well_count)
            if counter == well_count:
                break

        output = pd.DataFrame(self.table_dict)
        return output

    def _handle_vars(self, vars):
        if 'GOR' in vars:
            vars.loc[vars['GOR'] < 0.01, 'GOR'] = 0.7
            vars.loc[vars['oil'] < 0.001, 'oil'] = 0.001
            vars.loc[vars['gas'] < 0.001, 'gas'] = 0.001
            vars['GOR_residuals'] = np.log10(vars['GOR']) - \
                                    np.log10(vars['gas']/vars['oil'])
            self.var_stats['gor_stats'] = vars['GOR'].describe()
            self.var_stats['gor_res_stats'] = vars['GOR_residuals'].describe()

        if 'log_inferred_oil_from_nearest_producer' in vars:
            self.var_stats['oil_stats'] = vars[
                'log_inferred_oil_from_nearest_producer'].describe()
            vars['log_res_oil'] = vars[
                'log_inferred_oil_from_nearest_producer'] - \
                                  vars['log_oil_rate']
            self.var_stats['oil_res_stats'] = vars['log_res_oil'].describe()
            self.current_field['Max Rate'] = 10**self.var_stats[
                                                 'oil_stats']['max'] * 1.5
            self.current_field['P25 Rate'] = 10**self.var_stats[
                                                 'oil_stats']['75%'] * 1.5
            self.current_field['P50 Rate'] = 10**self.var_stats[
                                                 'oil_stats']['50%'] * 1.5
            self.current_field['P75 Rate'] = 10**self.var_stats[
                                                 'oil_stats']['25%'] * 1.5

        if 'log_inferred_gas_from_nearest_producer' in vars:
            self.var_stats['gas_stats'] = vars[
                'log_inferred_gas_from_nearest_producer'].describe()
            vars['log_res_gas'] = vars[
                                     'log_inferred_gas_from_nearest_producer']\
                                  - vars['log_gas_rate']
            self.var_stats['gas_res_stats'] = vars['log_res_gas'].describe()

        if 'log_inferred_water_from_nearest_producer' in vars:
            self.var_stats['water_stats'] = vars[
                'log_inferred_water_from_nearest_producer'].describe()
            vars['log_res_water'] = vars[
                'log_inferred_water_from_nearest_producer']\
                - vars['log_water_rate']
            self.var_stats['water_res_stats'] = vars['log_res_water'].\
                describe()

        # max completion height + 1000
        if 'Completion_Height' in vars:
            gross_rate_per_ft = vars['NearestProducerOilPerFt'] + \
                                vars['NearestProducerWaterRatePerFt']
            self.var_stats['gross_per_ft_stats'] = gross_rate_per_ft.describe()
            self.var_stats['completion_height_stats'] = vars[
                'Completion_Height'].describe()
            # completion height linear func + normalized residual

        # est remaining oil
        if 'Est_Remaining_Oil' in vars:
            self.var_stats['remaining_oil_stats'] = vars[
                'Est_Remaining_Oil'].describe()

        # est HAFWL
        if 'Est_HAFWL' in vars:
            self.var_stats['fwl_stats'] = vars['Est_HAFWL'].describe()
            if self.current_field['FWL'] > vars['z'].describe()['25%']:
                self.current_field['FWL'] = vars['z'].describe()['25%'] - 300

        if 'KH' in vars:
            self.var_stats['KH'] = vars['KH'].describe()
            if np.isnan(self.var_stats['KH']['max']):
                self.var_stats['KH']['min'] = 0.01
                self.var_stats['KH']['mean'] = 20
                self.var_stats['KH']['std'] = 10
            else:
                self.current_field['KH_MAX'] = self.var_stats['KH']['max'] * 1.5


    def _handle_output(self, x):
        process = True
        if 'Max Rate' in self.current_field:
            if x['gross_rate'] > self.current_field['Max Rate']:
                process = False
            else:
                process = self._handle_percentiles(x['gross_rate'])
        if process:
            if len(self.table_dict) == 0:
                self.table_dict = x
            else:
                for key, value in x.items():
                    if not isinstance(self.table_dict[key], list):
                        self.table_dict[key] = [self.table_dict[key]]
                    self.table_dict[key].append(value)
            return True
        return False

    def _handle_percentiles(self, x_rate)->bool:
        if x_rate >= self.current_field['P25 Rate']:
            output, self.p1 = self.update_var_and_set_output(self.p1, self.well_count)
        # elif self.current_field['P90 Rate'] > x_rate:
        #     output, self.p90 = self.update_var_and_set_output(self.p90, self.well_count)
        elif self.current_field['P75 Rate'] > x_rate:
            output, self.p75 = self.update_var_and_set_output(self.p75, self.well_count)
        elif self.current_field['P50 Rate'] > x_rate:
            output, self.p50 = self.update_var_and_set_output(self.p50, self.well_count)
        elif self.current_field['P25 Rate'] > x_rate:
            output, self.p25 = self.update_var_and_set_output(self.p25, self.well_count)
        # elif self.current_field['P10 Rate'] > x_rate:
        #     output, self.p10 = self.update_var_and_set_output(self.p10, self.well_count)
        else:
            output = False
        return output

    @staticmethod
    def update_var_and_set_output(counter_val, well_count)->bool:
        counter_val += 1
        if counter_val > int(well_count/3):
            return False, counter_val
        else:
            return True, counter_val

    @staticmethod
    def _progress_tracker(counter, well_count):
        if counter % 100 == 0:
            print('Pct done: {}'.format(counter / well_count))

    def return_synthetic_data(self, field=None):
        # Get field characteristics
        while True:
            self.output = defaultdict(list)
            if field is None:
                self.get_field()
            else:
                self.output['field'] = 0
                self.current_field = field
                self.output['lithology'] = field['lithology']
                self.output['viscosity'] = field['viscosity']
            self.compute_porosity_perm_attributes()
            self.compute_orignal_sw_and_height()
            self.compute_recoveries_at_radiuses()
            self.output['KH'] = self.output['True_KH'] * (
                        random.randint(60, 140) * 0.01)
            self.compute_injection_rate_and_vrr()
            self.adjust_sw_for_injection()
            self.compute_pressure()
            self.compute_relative_perms()
            self.compute_drainage_radius()
            self.compute_viscosity_and_skin()
            self.compute_surface_rates()
            self.compute_nearest_neighbor_rates()
            self.compute_estimated_recovery_factor()
            self.compute_bucketed_porosity()
            self.compute_weighted_sw()
            self.compute_volumes_in_place()
            self.compute_eur()
            self.output['inferred_oil_from_nearest_producer'] = \
                self.output['NearestProducerOilPerFt'] * \
                self.output['Completion_Height']
            self.output['inferred_water_from_nearest_producer'] = \
                self.output['NearestProducerWaterRatePerFt'] * \
                self.output['Completion_Height']
            self.output['inferred_gas_from_nearest_producer'] = \
                self.output['NearestProducerGasRatePerFt'] * \
                self.output['Completion_Height']
            yield self.output

    def compute_volumes_in_place(self):
        self.output['True_OOIP'] = self.output['True_Drainage_R'] * \
                                   self.output['True_Drainage_R'] * \
                                   math.pi * (1 - self.output[
            'True_Original_SW'] - self.output['True_Original_SG']) * \
                                   self.output[
                                       'True_PV'] * 2.29569e-5 * 7758.3
        self.output['True_OGIP'] = self.output['True_Drainage_R'] * \
                                   self.output['True_Drainage_R'] * \
                                   math.pi * (
                                       self.output['True_Original_SG']) * \
                                   self.output['True_PV'] * 0.001

    def compute_eur(self):
        delta_recovery_4000 = self.output[
                                  'True_Recovery_Expected_Recovery_Oil_4000'] - \
                              self.output['Current_Recovery_Oil_4000']
        delta_recovery_500 = self.output[
                                 'True_Recovery_Expected_Recovery_Oil_500'] - \
                             self.output['Current_Recovery_Oil_500']
        delta_recovery = (delta_recovery_4000 + delta_recovery_500) / 2
        if self.output['oil_rate'] > 0:
            self.output['True_EUR_oil'] = self.output['True_OOIP'] * (
                    delta_recovery + random.uniform(-0.05, 0.05))
        else:
            self.output['True_EUR_oil'] = 0
        if (self.output['free_gas_rate'] + self.output[
            'solution_gas_rate']) > 0:
            self.output['True_EUR_gas'] = self.output['True_OGIP'] * (
                    delta_recovery + random.uniform(-0.05, 0.05)) + \
                                          self.output['True_EUR_oil'] * \
                                          self.output['GOR']
        else:
            self.output['True_EUR_gas'] = 0
        if self.output['True_EUR_oil'] < 0:
            self.output['oil_rate'] = 0
            self.output['True_EUR_oil'] = 0
        if self.output['True_EUR_gas'] < 0:
            self.output['free_gas_rate'] = 0
            self.output['solution_gas_rate'] = 0
            self.output['True_EUR_gas'] = 0

    def compute_weighted_sw(self):
        self.output['Weighted_Sw_est'] = random.uniform(-0.3, 0.3) + \
                                         self.output[
                                             'True_SW']  # TODO Check
        # limits
        if self.output['Weighted_Sw_est'] < 0.05:
            self.output['Weighted_Sw_est'] = random.uniform(-0.025,
                                                            0.1) + 0.05
        if self.output['Weighted_Sw_est'] > 1.0:
            self.output['Weighted_Sw_est'] = random.uniform(-0.1,
                                                            0.1) + 0.9


    def compute_bucketed_porosity(self):
        # Bucketed Porosity
        for num in range(0, 60):
            self.output['porosity_{}'.format(num)] = 0
        pore_vol_budget = self.output['True_PV'] * random.randint(80,
                                                                  120) * \
                          0.01
        avg_poro = pore_vol_budget / self.output['Completion_Height']
        for x in range(0, int(self.output['Completion_Height'])):
            poro = random.normalvariate(avg_poro, 0.1)
            while poro > self.current_field['max_porosity'] or poro < 0:
                poro = random.normalvariate(avg_poro, 0.1)
            bucket_num = int(poro * 100)
            self.output['porosity_{}'.format(bucket_num)] += 1

    def compute_estimated_recovery_factor(self):
        # Recovery % oil and gas
        self.output['Est_500ft_Recovery'] = self.output[
                                                'Current_Recovery_Oil_500'] \
                                            + np.random.normal(0.0, 0.05)
        self.output['Est_4000ft_Recovery'] = self.output['Est_500ft_Recovery']
        self.output['Est_Remaining_Oil'] = max([0.001, self.output[
            'True_Remaining_Oil'] + np.random.normal(-0.03, 0.015)])

    def compute_drainage_radius(self):
        self.output['NearestProducerDistance'] = random.randint(50, 8000)
        if self.output['NearestProducerDistance'] > 400:
            self.output['True_Drainage_R'] = (self.output[
                                                  'NearestProducerDistance']
                                              / 2) * random.randint(
                60, 140) * 0.01
        else:
            self.output['True_Drainage_R'] = (self.output[
                                                  'NearestProducerDistance']
                                              / 2) * random.randint(
                60, 140) * 0.01 + \
                                             (400 - self.output[
                                                 'NearestProducerDistance']
                                              / 4) * \
                                             random.randint(60, 140) * 0.01

    def compute_nearest_neighbor_rates(self):
        oil = self.zero_check(self.output['gross_rate'] *
                              (self.output['oil_cut']))
        log_oil_per_ft = math.log10(self.zero_check(self.output['oil_per_ft']))
        log_oil = math.log10(oil)
        water = self.zero_check(self.output['gross_rate']
                                * (1 - self.output['oil_cut']))
        log_water = math.log10(water)
        try:
            log_gor = math.log10(self.output['Actual_GOR'])
        except Exception as ex:
            log_gor = 1
            print(ex)
        gor_residual = np.random.normal(
            self.var_stats['gor_res_stats']['mean'],
            self.var_stats['gor_res_stats']['std'])
        neighbor_gor = 10**(log_gor+gor_residual)
        solution_gas = oil * neighbor_gor
        gas = self.zero_check(self.output['free_gas_rate']
                              + solution_gas)
        log_gas = math.log10(gas)
        log_error_random_factor_oil = np.random.normal(
            loc=self.var_stats['oil_res_stats']['50%'],
            scale=(self.var_stats['oil_res_stats']['75%'] -
                  self.var_stats['oil_res_stats']['25%'])/2)
        log_error_random_factor_gas = np.random.normal(
            loc=self.var_stats['gas_res_stats']['50%'],
            scale=(self.var_stats['gas_res_stats']['75%'] -
                  self.var_stats['gas_res_stats']['25%'])/2)
        log_error_random_factor_water = np.random.normal(
            loc=self.var_stats['water_res_stats']['50%'],
            scale=(self.var_stats['water_res_stats']['75%'] -
                  self.var_stats['water_res_stats']['25%'])/2)
        log_gross = math.log10(oil + water)
        log_gross = math.log10(oil + water)

        log_water_per_ft = math.log10(self.zero_check(
            self.output['water_per_ft']))
        log_gas_per_ft = math.log10(self.zero_check(self.output['gas_per_ft']))

        self.output['log_oil_rate'] = log_oil
        self.output['log_water_rate'] = log_water
        self.output['log_gas_rate'] = log_gas
        self.output['log_gross_rate'] = log_gross

        self.output['NearestProducerLogOilRate'] = \
            log_oil + log_error_random_factor_oil
        self.output['NearestProducerLogWaterRate'] = \
            log_water + log_error_random_factor_water
        self.output['NearestProducerLogGasRate'] = \
            log_gas + log_error_random_factor_gas
        self.output['NearestProducerLogGrossRate'] = \
            math.log10(10**self.output['NearestProducerLogOilRate'] +
                  10**self.output['NearestProducerLogWaterRate'])
        self.output['NearestProducerLogOilPerFt'] = \
            log_oil_per_ft + log_error_random_factor_oil
        self.output['NearestProducerLogWaterPerFt'] = \
            log_water_per_ft + log_error_random_factor_oil
        self.output['NearestProducerLogGasPerFt'] = \
            log_gas_per_ft + log_error_random_factor_oil
        self.output['NearestProducerGrossRate'] = 10**(
            self.output['NearestProducerLogGrossRate'])
        self.output['NearestProducerOilRate'] = 10**(
            self.output['NearestProducerLogOilRate'])
        self.output['NearestProducerWaterRate'] = 10**(
                self.output['NearestProducerLogWaterRate'])
        self.output['NearestProducerGasRate'] = 10**(
                self.output['NearestProducerLogGasRate'])
        self.output['NearestProducerOilPerFt'] = 10 ** (
            self.output['NearestProducerLogOilPerFt'])
        self.output['NearestProducerWaterRatePerFt'] = 10 ** (
            self.output['NearestProducerLogWaterPerFt'])
        self.output['NearestProducerGasRatePerFt'] = 10 ** (
            self.output['NearestProducerLogGasPerFt'])
        self.output['GOR'] = 10**self.output['NearestProducerLogGasRate'] / \
            10**self.output['NearestProducerLogOilRate']

    @staticmethod
    def zero_check(val):
        if val < 0.001:
            val = 0.001
        return val

    def compute_surface_rates(self):
        drainage_skin_factor = math.log10(
            self.output['True_Drainage_R'] / (5.5 / 12)) + 0.75 + \
                               self.output['skin']
        denominator = self.visocity_volume_fac * drainage_skin_factor
        denominator_gas = self.visocity_volume_fac_gas * drainage_skin_factor
        self.output['oil_rate'] = (self.output['True_Pressure'] *
                                   self.output['oil_kh']) / denominator
        self.output['water_rate'] = (self.output['True_Pressure'] *
                                     self.output['water_kh']) / denominator
        self.output['free_gas_rate'] = ((self.output['True_Pressure'] *
                                         self.output[
                                             'gas_kh']) /
                                        denominator_gas) * 0.01
        while True:
            if self.current_field['Bubble Point'] > self.output['True_Pressure']:
                gor = np.random.uniform(
                    self.var_stats['gor_stats']['50%'],
                    (self.var_stats['gor_stats']['75%']-
                    self.var_stats['gor_stats']['50%'])/2)
            else:
                gor = np.random.normal(
                    self.var_stats['gor_stats']['25%'],
                    self.var_stats['gor_stats']['50%'] -
                    self.var_stats['gor_stats']['25%'])
            if gor > 0:
                break
        self.output['solution_gas_rate'] = self.output['oil_rate'] * gor
        self.adjust_rates_for_lift_type()
        self.output['gross_rate'] = self.output['oil_rate'] + self.output[
            'water_rate']
        if self.output['gross_rate'] > 0:
            self.output['oil_cut'] = self.output['oil_rate'] / self.output[
                'gross_rate']
        else:
            self.output['oil_cut'] = 0
        if self.output['oil_rate'] > 0:
            self.output['Actual_GOR'] = self.output['solution_gas_rate'] / \
                                 self.output['oil_rate']
        else:
            self.output['Actual_GOR'] = 0.0000000001
        self.output['oil_per_ft'] = self.output['oil_rate']/\
                                    self.output['Completion_Height']
        self.output['water_per_ft'] = self.output['water_rate'] / \
                                    self.output['Completion_Height']
        self.output['gas_per_ft'] = (self.output['solution_gas_rate'] +
                                     self.output['free_gas_rate']) / \
                                    self.output['Completion_Height']
        self.output['gross_per_ft'] = self.output['gross_rate'] / \
                                      self.output['Completion_Height']
        if self.output['gross_per_ft'] > \
                self.var_stats['gross_per_ft_stats']['max']:
            self.scale_vars()

    def scale_vars(self):
        while True:
            new_gross_per_ft = np.random.normal(
                self.var_stats['gross_per_ft_stats']['50%'],
                (self.var_stats['gross_per_ft_stats']['75%'] -
                self.var_stats['gross_per_ft_stats']['25%'])/2
            )
            if new_gross_per_ft > 0:
                break
        scaling_factor = new_gross_per_ft / self.output['gross_per_ft']
        for key, item in self.output.items():
            skip_keys = ('lift_type', 'oil_cut', 'Actual_GOR', 'field',
                         'lithology', 'viscosity', 'Completion_Height',
                         'True_HAFWL', 'True_Original_SW', 'z',
                         'True_RelativeGOC', 'True_Original_SG',
                         'Est_HAFWL', 'Est Relative GOC',
                         'True_Recovery_Expected_Recovery_Oil_500',
                         'True_Recovery_Expected_Recovery_Oil_4000',
                         'Current_Recovery_Oil_500',
                         'Current_Recovery_Oil_4000',
                         'True_Remaining_Oil', 'NearestInjectorDistance',
                         'InjectorType', 'VRR', 'NearestInjectorWaterRate',
                         'NearestInjectorGasRate', 'True_SW', 'True_SG',
                         'True_Pressure', 'est_pressure',
                         'NearestProducerDistance', 'True_Drainage_R', 'skin',
                         )
            if key in skip_keys:
                continue
            self.output[key] = item * scaling_factor

    def adjust_rates_for_lift_type(self):
        lift_type = random.randint(0, 2)
        # 0 = rod well
        # 1 = ESP/GAS Lift
        # 2 = FLOWING
        gross_fluid_raw = self.output['oil_rate'] + self.output['water_rate']
        adj_factor = 1
        if lift_type == 0:
            if gross_fluid_raw > 800:
                adj_factor = 800 / gross_fluid_raw
        elif lift_type == 1:
            if gross_fluid_raw > 4000:
                adj_factor = 800 / gross_fluid_raw
        elif lift_type == 2:
            adj_factor = 0.5
        self.output['oil_rate'] = self.output['oil_rate'] * adj_factor
        self.output['water_rate'] = self.output['water_rate'] * adj_factor
        self.output['solution_gas_rate'] = self.output[
                                        'solution_gas_rate'] * adj_factor
        self.output['lift_type'] = lift_type


    def compute_relative_perms(self):
        self.output['oil_kh'] = self.compute_rel_perm_oil(
            self.output['True_SW'], self.output['True_SG'],
            self.output['True_KH'])
        self.output['water_kh'] = self.compute_rel_perm_water(
            self.output['True_SW'], self.output['True_SG'],
            self.output['True_KH'])
        self.output['gas_kh'] = self.compute_rel_perm_gas(
            self.output['True_SW'], self.output['True_SG'],
            self.output['True_KH'])

    def compute_pressure(self):
        self.output['True_Pressure'] = self.compute_true_pressure(
            self.output['Current_Recovery_Oil_4000'],
            self.output['NearestInjectorDistance'],
            self.output['VRR'], self.current_field['discovery_pressure'])
        self.output['est_pressure'] = self.output['True_Pressure'] * (
                random.randint(80, 120) * 0.01)

    def adjust_sw_for_injection(self):
        self.output['True_SW'] = self.compute_current_sw(
            self.injector_rate, self.output['InjectorType'],
            self.output['NearestInjectorDistance'],
            self.output['Current_Recovery_Oil_500'],
            self.current_field['has_aquifer'], self.output['True_Original_SW'])
        self.output['True_SG'] = self.compute_sg(
            self.output['NearestInjectorDistance'],
            self.output['InjectorType'], self.output['z'], self.current_field['GOC'],
            self.output['True_SW'])

    def compute_injection_rate_and_vrr(self):
        self.output['NearestInjectorDistance'] = random.randint(20, 8000)
        self.output['InjectorType'] = random.randint(0, 2)
        if self.output['InjectorType'] == 1:
            water_rate = random.randint(0, 20000)
            gas_rate = 0
            self.injector_rate = water_rate
            self.output['VRR'] = random.uniform(0.0, 5.0)
        elif self.output['InjectorType'] == 0:
            water_rate = 0
            gas_rate = random.randint(0, 20000)
            self.injector_rate = gas_rate
            self.output['VRR'] = random.uniform(0.0, 5.0)
        else:
            water_rate = 0
            gas_rate = 0
            self.injector_rate = 0
            if self.current_field['has_aquifer']:
                self.output['VRR'] = random.uniform(0.0, 5.0)
            else:
                self.output['VRR'] = 0
        self.output['NearestInjectorWaterRate'] = water_rate
        self.output['NearestInjectorGasRate'] = gas_rate

    def compute_viscosity_and_skin(self):
        self.output['skin'] = random.randint(-1, 20)
        self.visocity_volume_fac = (self.current_field['viscosity'] * 141.2 * 1.02)
        self.visocity_volume_fac_gas = (1.29 * 200)

    def compute_recoveries_at_radiuses(self):
        self.output['True_Recovery_Expected_Recovery_Oil_500'] = \
            np.random.normal(self.current_field['expected_recovery'], 0.01)
        self.output['Current_Recovery_Oil_500'] = np.random.uniform(
            0.001,
            self.output['True_Recovery_Expected_Recovery_Oil_500'])
        self.output['True_Recovery_Expected_Recovery_Oil_4000'] = \
            self.output['True_Recovery_Expected_Recovery_Oil_500']
        self.output['Current_Recovery_Oil_4000'] =\
            self.output['Current_Recovery_Oil_500']
        self.output['True_Remaining_Oil'] = 1 -\
            (self.output['Current_Recovery_Oil_4000'] /
            self.output['True_Recovery_Expected_Recovery_Oil_4000'])

    def compute_orignal_sw_and_height(self):
        self.output['True_HAFWL'] = random.randint(0, 4000)
        self.output['True_Original_SW'] = self.compute_original_sw(
            self.current_field,
            self.output['True_PV'] / self.output['Completion_Height'],
            self.output['True_KH'] / self.output['Completion_Height'],
            self.output['True_HAFWL'])
        self.output['z'] = self.current_field['FWL'] +\
                           self.output['True_HAFWL']
        self.output['True_RelativeGOC'] = (
            self.output['True_HAFWL'] +
            self.current_field['FWL']) - self.current_field['GOC']
        if self.output['True_RelativeGOC'] > 0:
            self.output['True_Original_SG'] = 1 - self.output[
                'True_Original_SW']
        else:
            self.output['True_Original_SG'] = 0
        self.output['Est_HAFWL'] = self.output[
                                       'True_HAFWL'] + random.randint(-300,
                                                                      300)
        self.output['Est Relative GOC'] = self.output[
                                              'True_RelativeGOC'] + \
                                          random.randint(
                                              -150, 150)

    def compute_porosity_perm_attributes(self):
        if 'KH_Max' in self.current_field:
            kh_max = self.current_field['KH_Max']
        else:
            kh_max = 1000000
        self.output['True_KH'] = np.random.uniform(
            low=self.var_stats['KH']['min']+0.001, high=kh_max)
        self.output['Completion_Height'], self.output['perm_avg_ft'], \
        self.output['True_KH'] = self.get_completion_height(
            self.output['True_KH'], self.current_field['lithology'])
        self.output['True_KH'] = self.regulate_kh_and_completion_height(
            self.output['Completion_Height'], self.output['perm_avg_ft'],
            self.output['True_KH'])
        self.output['avg_porosity_ft'] = self.perm_to_porosity(
            self.current_field['hperm'], self.current_field['jperm'],
            self.output['perm_avg_ft']) + \
                                         random.uniform(-0.1, 0.1)
        if self.output['avg_porosity_ft'] < 0:
            self.output['avg_porosity_ft'] = random.uniform(0.001, 0.05)

        self.output['True_PV'] = self.output['avg_porosity_ft'] * \
                                 self.output['Completion_Height']

    def get_field(self):
        field_int = random.randint(0, self.no_of_fields - 1)
        if field_int in self.fields.keys():
            field = self.fields[field_int]
        else:
            # generate field
            field = next(self.field_gen)
            self.fields[field_int] = field
        self.output['field'] = field_int
        self.current_field = field
        self.output['lithology'] = field['lithology']
        self.output['viscosity'] = field['viscosity']

    @staticmethod
    def compute_original_sw(current_field, avg_por, avg_perm, hafwl):
        a = current_field['J_A']
        b = current_field['J_B']
        c = current_field['J_C']
        if hafwl == 0:
            hafwl = 0.00001
        sw = a * ((avg_perm / avg_por) ** (0.5) * hafwl) ** b + c
        if sw > 1:
            sw = 1
        elif sw < 0.05:
            sw = 0.05
        return sw

    def compute_true_pressure(self, recovery, injection_distance, VRR,
                              starting_pressure):
        if VRR < 0.001:
            pressure = starting_pressure / (1 + 2 * recovery)
        else:
            pressure = starting_pressure / (1 + 2 * (
                    recovery - (min((VRR, 1)) * recovery)))
        return pressure

    def compute_current_sw(self, injection_amt, injection_type, distance,
                           recovery, has_aquifer, orig_sw):
        # injection_amt_factor = self.ceiling_val(injection_amt / 2000)
        # injection_distance_factor = self.zero_floor(1 - distance / 2000)
        # if injection_amt < 10 and has_aquifer:
        #     effective_recovery = recovery
        # elif injection_amt < 10:
        return (1-orig_sw) * recovery + orig_sw
        # else:
        #     effective_recovery = (0.10 + random.uniform(-0.025, 0.025)) * \
        #                          injection_amt_factor * \
        #                          injection_distance_factor + recovery
        # current_factor = 1 / (1 + math.e ** (1 - (4 * effective_recovery)))
        # if injection_type == 0 and injection_amt > 10:
        #     return 1 - current_factor  # Gas injectors reduce Sw
        # else:
        #     return current_factor  # Aquifers/Water injecters increase Sw

    def compute_rel_perm_oil(self, sw, sg, perm_total):
        sw *= 100
        sg *= 100
        oil_gas_perm = ((-0.0000004 * sg) ** 4 + (-0.00006 * sg) ** 3 + (
                    0.0032 * sg) ** 2 +
                        (-0.0872 * sg) + 1)
        if oil_gas_perm < 0:
            oil_gas_perm = 0
        oil_water_perm = (-0.000000442 * sw) ** 4 + (0.0000695 * sw) ** 3 + (
                    -0.0033 * sw) ** 2 + \
                         (-0.0243 * sw) + 1.45
        if oil_water_perm > 0.9:
            oil_water_perm = 0.9
        if oil_gas_perm > 0.9:
            oil_gas_perm = 0.9
        if sg > 0:
            rel_perm = oil_gas_perm * oil_water_perm
        else:
            rel_perm = oil_water_perm
        if rel_perm < 0:
            rel_perm = 0
        return rel_perm * perm_total

    def compute_rel_perm_water(self, sw, sg, perm_total):
        sw = sw * 100
        sg = (1 - sg) * 100
        oil_gas_perm = (-0.0000000325 * sg) ** 4 + (0.00000668 * sg) ** 3 + (
                    -0.000379 * sg) ** 2 + \
                       (0.00951 * sg) + 0.0485
        oil_water_perm = ((-0.0000001 * sw) ** 4 + (0.00002 * sw) ** 3 + (
                    -0.001 * sw) ** 2 + \
                          (0.0294 * sw) + -0.2691)
        if oil_gas_perm < 0:
            oil_gas_perm = 0
        if oil_gas_perm > 0.8:
            oil_gas_perm = 0.8
        if oil_water_perm < 0:
            oil_water_perm = 0
        elif oil_water_perm > 0.8:
            oil_water_perm = 0.8
        if sg > 0:
            rel_perm = oil_gas_perm * oil_water_perm
        else:
            rel_perm = oil_water_perm
        return rel_perm * perm_total

    def compute_rel_perm_gas(self, sw, sg, perm_total):
        if sg > 0:
            relperm = 1 / (1 + math.e ** (4 + -80 * sg ** 3))
            relperm = self.ceiling_val(relperm, ceiling=0.70)
        else:
            relperm = 0
        return relperm * perm_total

    def compute_porosity_from_perm(self, a, m, perm, fperm=7 * 10 ** 6,
                                   gperm=4.5):
        porosity_val = (a / ((fperm / perm) ** (1 / gperm))) ** (1 / m)
        return porosity_val

    @staticmethod
    def zero_floor(val):
        if val < 0:
            return 0
        return val

    @staticmethod
    def ceiling_val(val, ceiling=1.0):
        if val > 1:
            return 1
        return val

    @staticmethod
    def perm_transform(hperm, jperm, porosity):
        perm = 10 ** (hperm * porosity + jperm)
        if perm > 20000:
            perm = 20000
        return perm

    @staticmethod
    def perm_to_porosity(hperm, jperm, perm):
        porosity = ((perm ** (1 / 10)) - jperm) / hperm
        if porosity > 0.38:
            porosity = 0.38
        elif porosity < 0:
            porosity = 0.01
        return porosity

    def get_completion_height(self, kh, lithology):
        if lithology == 'sandstone':
            perm_avg_ft = random.randint(1, 1000) * 1.0
        elif lithology == 'mixed_porc_sand':
            perm_avg_ft = random.randint(1, 25) * 1.0
        elif lithology == 'porc':
            perm_avg_ft = random.randint(1, 2) * 0.01
        elif lithology == 'unconsolidated_sand':
            perm_avg_ft = random.randint(500, 2000) * 1.0
        completion_height = kh / perm_avg_ft
        if 'Comp Height Cap' in self.current_field:
            completion_height_limit = self.current_field['Comp Height Cap']
        else:
            completion_height_limit = 4000
        if completion_height > completion_height_limit:
            completion_height = completion_height_limit + random.randint(-500, 500)
            if completion_height <= 0:
                completion_height = random.randint(1, completion_height_limit)
            kh = completion_height * perm_avg_ft
        return completion_height, perm_avg_ft, kh

    @staticmethod
    def regulate_kh_and_completion_height(height, kh_avg, kh):
        if height > 2000:
            kh = kh_avg * (2000 + random.randint(-400, 400))
        return kh

    @staticmethod
    def check_seed(seed):
        if seed is not None:
            random.seed = seed

    @staticmethod
    def compute_sg(injector_distance, injector_type, z, goc, sw):
        if z > goc:
            return 1 - sw
        if injector_type == 0:
            if injector_distance > 2000:
                return 0.01
            else:
                sg = 1 - injector_distance / 2000 * 0.15 + 0.01
                if (sw + sg) > 1:
                    sg = 1 - sw
                return sg
        else:
            return 0.0
