# -*- coding: utf-8 -*-

# from setuptools import setup
# from distutils.core import setup
from setuptools import setup

setup(name='crcdal',
      version='0.1.1',
      description='Replication Code for Nathaniel Jones FORCE 2019 Talk on procedural generation of well feature data',
      url='https://github.com/CRC-IT/2019_FORCE_talk',
      author='CRC BDA Team',
      author_email='nathan.geology@gmail.com',
      license='CC 2.0 BY',
      packages=['syn_inflow'],
      install_requires=[  # 'cx_Oracle',
          'pandas',
          'numpy',
          # 'pyodbc',
          'scipy',
          # 'patsy',
          'sqlalchemy',
          'joblib',
          'scikit-learn',
          'torch',
          # 'xgboost',
          # 'progress',
          'scikit-image',
          'imblearn',
          # 'geopandas',
          # 'altair',
          'numba',
          'psutil',
          'lasio',
          'matplotlib',
          'pyepsg',
      ],
      zip_safe=False,
      test_suite='nose.collector',
      tests_require=['nose'],
      package_data={'syn_inflow': ['package_data/*.csv',
                                   'package_data/*.pkl',
                                   'packageData/database_name_cleaner_tables/*.csv'
                               ]})
