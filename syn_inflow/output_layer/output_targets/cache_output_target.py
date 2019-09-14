
from .output_target import OutputTarget
from syn_inflow.data_science_layer.utilities.cache_path import get_cache_path
import pkg_resources


class CacheOutputTarget(OutputTarget):
    path = None

    @classmethod
    def write_to_target(cls, data, name):
        self = cls()
        # Get Configuration folder + subfolder
        self.path = self.get_cache_path()
        # Put Data On the Bucket (with format)
        func = self.available_formats[self.format]
        func(data, self.path + name + '.' + self.format)
        return self.path + name + '.' + self.format

    @staticmethod
    def get_cache_path():
        subfolder = get_cache_path()
        path = pkg_resources.resource_filename(
            'syn_inflow', '/cache/' + subfolder + '/')
        pkg_resources.ensure_directory(path)
        return path
