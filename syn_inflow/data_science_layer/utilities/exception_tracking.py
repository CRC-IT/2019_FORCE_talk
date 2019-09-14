
from .singleton import Singleton
import traceback as tb
from collections import defaultdict
import datetime as dttm
import pandas as pd
import pkg_resources
import uuid

class ExceptionTracking(object, metaclass=Singleton):
    log = defaultdict(list)
    script_run_name = ''

    @classmethod
    def send_exception(cls,
                       info,
                       tag,
                       ds_id  # Data Struct ID
                       ):
        obj = cls()
        obj.log_exception(info, tag, ds_id)

    def log_exception(self,
                      info,
                      tag,
                      ds_id  # Data Struct ID
                      ):
        self.log['info'].append(info)
        self.log['tag'].append(tag)
        self.log['ds_id'].append(ds_id)
        self.log['exception_string'].append(tb.format_exc().replace('\n', ''))

    def thread_exception(self, info,
                        tag,
                        ds_id,  # Data Struct ID
                        path):
        thread = defaultdict(list)
        thread['info'].append(info)
        thread['tag'].append(tag)
        thread['ds_id'].append(ds_id)
        thread['exception_string'].append(tb.format_exc().replace('\n', ''))
        self._save_thread(path, thread)

    def reset(self, save=False, path=None):
        if save:
            self._save(path)
        self.log.clear()
        self.script_run_name = ''

    def _save(self, path):
        if path is None:
            path = self.get_default_path()
        date_tag = self.date_tag()
        path = path + self.script_run_name + '_' + \
               date_tag + '.txt'
        table = self.generate_report_table(self.log)
        table.to_csv(path, sep='*', index=False)

    def _save_thread(self, path, thread_exception):
        if path is None:
            path = self.get_default_path()
        date_tag = self.uuid_tag()
        path = path + self.script_run_name + '_' + \
               date_tag + '.txt'
        table = self.generate_report_table(thread_exception)
        table.to_csv(path, sep='*', index=False)

    @staticmethod
    def generate_report_table(log) -> pd.DataFrame:
        """"""
        table = pd.DataFrame(log)
        index = ['ds_id', 'tag', 'info', 'exception_string']
        table = table.reindex(index, axis='columns')
        return table

    @staticmethod
    def get_default_path() -> str:
        path = pkg_resources.resource_filename(
            'syn_inflow', 'cache/exception_reports/')
        pkg_resources.ensure_directory(path)
        return path

    @staticmethod
    def date_tag():
        now = dttm.datetime.now()
        return str(now.microsecond) + str(now.second) + str(now.minute) + str(now.hour) + str(now.day) \
               + str(now.month) + str(now.year)

    @staticmethod
    def uuid_tag():
        return str(uuid.uuid4())