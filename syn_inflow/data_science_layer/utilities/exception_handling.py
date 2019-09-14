"""
Authored by Nathaniel Jones, 
Modified and maintained by the Big Data Analytics Team
Copyright California Resources Corporation 2018, all rights reserved
"""
from .exception_tracking import ExceptionTracking
import functools
import pkg_resources
import pandas as pd
import glob
import shutil


def function_raises_val(returns, exception_info, tag):
    """Decorator to provide standard error handling to a function"""

    def _function_raises_val(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as ex:
                print(ex)
                log_exception(args, kwargs, exception_info, tag)
                return returns

        return wrapper

    return _function_raises_val


def function_raises_val_for_thread(returns, exception_info, tag):
    """Decorator to provide standard error handling to a function in
    paralyzation"""

    def _function_raises_val_for_thread(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as ex:
                print(ex)
                path = ExceptionTracking().get_default_path() + \
                       'thread_exceptions/'
                pkg_resources.ensure_directory(path)
                save_thread_exception(args, kwargs, exception_info, tag, path)
                return returns

        return wrapper

    return _function_raises_val_for_thread


def function_concats_raised_vals_for_multi_thread(returns=None):
    """Decorator to provide standard error handling to a function in
    paralyzation"""

    def _function_concats_raised_vals_for_multi_thread(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                ret = func(*args, **kwargs)
                path = ExceptionTracking().get_default_path()
                threads_path = path + 'thread_exceptions/'
                all_files = glob.glob(threads_path + "/*.txt")
                for filename in all_files:
                    df = pd.read_csv(filename, sep='*', header=0,
                                     names=['ds_id', 'tag', 'info',
                                            'exception_string'])
                    ExceptionTracking().log['info'].append(df['info'][0])
                    ExceptionTracking().log['tag'].append(df['tag'][0])
                    ExceptionTracking().log['ds_id'].append(df['ds_id'][0])
                    ExceptionTracking().log['exception_string'].append(
                        df['exception_string'][0])
                shutil.rmtree(threads_path)
                return ret
            except Exception as ex:
                log_exception(args, kwargs,
                              'unexpected failure in multithreaded function',
                              'multithreaded func failure')
                return returns

        return wrapper

    return _function_concats_raised_vals_for_multi_thread


def throws_exception(exception_info='', tag='', throws=None):
    def _throws_exception(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as ex:
                print(ex)
                log_exception(args, kwargs, exception_info, tag)
                if throws is None:
                    raise ex
                else:
                    throws(str(ex))

        return wrapper

    return _throws_exception


def log_exception(args, kwargs, exception_info, tag):
    name = check_args(args, kwargs)
    if name == '':
        name = 'unknown'
    ExceptionTracking().log_exception(exception_info, tag, name)


def save_thread_exception(args, kwargs, exception_info, tag, path):
    name = check_args(args, kwargs)
    if name == '':
        name = 'unknown'
    ExceptionTracking().thread_exception(info=exception_info, tag=tag, ds_id=name, path=path)


def check_args(args, kwargs):
    output = check_args_for_ds_name(args)
    if output == '':
        output = check_kwargs_for_instance(kwargs)
    return output


def check_args_for_ds_name(args):
    if len(args) > 0:
        obj = args[0]
        if hasattr(obj, 'name'):
            return obj.name
    return ''


def check_kwargs_for_instance(kwargs):
    if 'interface' in kwargs:
        obj = kwargs['interface']
        return obj.name
    if 'well_name' in kwargs:
        return str(kwargs['well_name'])
    return ''
