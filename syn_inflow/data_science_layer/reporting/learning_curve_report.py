from .abstract_report import AbstractReport
from ..pipeline.abstract_pipline import AbstractPipeline
from ..utilities.cache_path import get_cache_path
import pandas as pd
import numpy as np
import pkg_resources
from collections import defaultdict
from sklearn.model_selection import learning_curve
from sklearn.metrics.scorer import make_scorer


class LearningCurveReport(AbstractReport):

    sub_folder = 'reports'
    cv_num = 3
    verbose = 0

    def report(self, pipeline: AbstractPipeline):

        train_examples = np.linspace(0.1, 1.0, 10)
        report_dict = defaultdict(list)

        for model in pipeline.get_models():
            model_name = model.short_name

            custom_scorer = make_scorer(model.scorer.score,
                                        greater_is_better=model.scorer.greater_is_better,
                                        needs_proba=model.scorer.needs_proba,
                                        needs_threshold=False)

            try:
                train_sizes, train_scores, test_scores = learning_curve(
                    model.best_model, pipeline.train, pipeline.train_y,
                    verbose=self.verbose, scoring=custom_scorer, cv=self.cv_num, train_sizes=train_examples, n_jobs=-1)
                report_dict['model_name'].append(model_name)
                report_dict['train_sizes'].append(train_sizes)
                report_dict['train_scores'].append(train_scores)
                report_dict['test_scores'].append(test_scores)
            except:
                print('Learning Curve Failed: ' + model_name)

        report_df = pd.DataFrame(report_dict)
        folder = get_cache_path()
        path = pkg_resources.resource_filename('crcdal', 'cache/' + folder + '/' + self.sub_folder +'/')
        pkg_resources.ensure_directory(path)
        report_df.to_csv(path + pipeline.dataset_tag + '_model_learning_curve_report.csv')
