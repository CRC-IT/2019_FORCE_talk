'''
Authored by Nathaniel Jones, 
Modified and maintained by the Big Data Analytics Team
Copyright California Resources Corporation 2018, all rights reserved
'''
from syn_inflow.data_science_layer.utilities.exception_tracking import ExceptionTracking
from syn_inflow.data_science_layer.utilities.singleton import Singleton


class ApiValidator(object, metaclass=Singleton):
    """ Class to help validate api numbers

    --- Public Methods
    validate_api14(*, api_list)
    validate_api12(*, api_list)
    validate_api10(*, api_list)
    """
    def validate_api(self, *, api_list, api_length):
        for idx, api in enumerate(api_list):
            try:
                api = self.__validate_for_all_api_types(api)
            except Exception as ex:
                ExceptionTracking().log_exception(
                    'Bad API', 'api_validation', api)
                continue

            if self.__check_api_is_not_equal_to(length=api_length, api=api):
                api = self.__correct_api_length(api=api, length=api_length)
            api_list[idx] = api
        return api_list

    def validate_api14(self, *, api_list):
        for idx, api in enumerate(api_list):
            try:
                api = self.__validate_for_all_api_types(api)
            except Exception as ex:
                ExceptionTracking().log_exception(
                    info='Bad API', tag='api_validation', ds_id=api)
                continue

            if self.__check_api_is_not_equal_to(length=14, api=api):
                api = self.__correct_api_length(api=api, length=14)
            api_list[idx] = api
        return api_list

    def validate_api12(self, *, api_list):
        for idx, api in enumerate(api_list):
            api = self.__validate_for_all_api_types(api)

            if self.__check_api_is_not_equal_to(length=12, api=api):
                api = self.__correct_api_length(api=api, length=12)
            api_list[idx] = api
        return api_list

    def validate_api10(self, *, api_list):
        for idx, api in enumerate(api_list):
            api = self.__validate_for_all_api_types(api)

            if self.__check_api_is_not_equal_to(length=10, api=api):
                api = self.__correct_api_length(api=api, length=10)
            api_list[idx] = api
        return api_list

    def __validate_for_all_api_types(self, api):
        if self.__check_is_not_str(api):
            api = self._convert_to_int(api)
            api = self.__convert_to_str(api)
        if self.__first_character_is_not_zero(api):
            api = self.__prepend_zero_to_api(api)
        self.__check_is_gibberish(api)
        return api

    def _convert_to_int(self, api):
        if isinstance(api, float):
            api = int(api)
        return api

    def __check_api_is_not_equal_to(self, *, length, api):
        if length == len(api):
            return False
        return True

    def __first_character_is_not_zero(self, api):
        if api[0] == '0':
            return False
        return True

    def __check_is_not_str(self, api):
        if type(api) == str:
            return False
        return True

    def __check_is_gibberish(self, api):
        if api[0:2] == '04':
            return False
        raise (ValueError('{api} is not a valid california API number'.format(api=api)))

    def __convert_to_str(self, api):
        return str(api)

    def __prepend_zero_to_api(self, api):
        return '0' + api

    def __correct_api_length(self, api, length):
        current_length = len(api)
        diff = length - current_length
        if diff > 0:
            api = self.__append_zeros_for_length(api, diff)
        else:
            api = api[0:length]
        return api

    def __append_zeros_for_length(self, api, num_zeros):
        for x in range(0, num_zeros):
            api = api + '0'
        return api
