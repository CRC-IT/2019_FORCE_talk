from crcdal.data_science_layer.reporting.abstract_report import AbstractReport
from crcdal.data_science_layer.pipeline.abstract_pipline import AbstractPipeline

import pandas as pd
import numpy as np
import pkg_resources
from crcdal.input_layer.configuration import Configuration
from ..utilities.cache_path import get_cache_path
from collections import defaultdict

from sklearn.model_selection import cross_val_score
from sklearn.metrics.scorer import make_scorer

class CrossValidationReport(AbstractReport):

    sub_folder = 'reports'
    cv_num = 3
    verbose = 0

    def report(self, pipeline: AbstractPipeline):

        report_dict = defaultdict(list)

        for model in pipeline.get_models():
            model_name = model.short_name

            custom_scorer = make_scorer(model.scorer.score,
                                        greater_is_better=model.scorer.greater_is_better,
                                        needs_proba=model.scorer.needs_proba,
                                        needs_threshold=False)

            try:
                cv = cross_val_score(
                    model.best_model, pipeline.train, pipeline.train_y,
                    verbose=self.verbose, scoring=custom_scorer, cv=self.cv_num, n_jobs=-1)
                report_dict['model_name'].append(model_name)
                report_dict['cross_val_score'].append(cv)
            except:
                print('Cross Val Failed: ' + model_name)

        report_df = pd.DataFrame(report_dict)
        folder = Configuration.get_cache_subfolder()
        path = pkg_resources.resource_filename('crcdal', 'cache/' + folder + '/' + self.sub_folder +'/')
        pkg_resources.ensure_directory(path)
        report_df.to_csv(path + pipeline.dataset_tag + '_model_cross_val_report.csv')
