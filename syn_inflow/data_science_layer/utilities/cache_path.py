import pkg_resources


def get_cache_path():
    path = pkg_resources.resource_filename('syn_inflow', '/cache')
    pkg_resources.ensure_directory(path)
    return path
