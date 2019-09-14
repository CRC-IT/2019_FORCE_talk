from .abstract_report import AbstractReport
from ..pipeline.abstract_pipline import AbstractPipeline
from ..scoring.mean_absolute_error_scorer import MeanAbsoluteErrorScorer
from ..utilities.cache_path import get_cache_path
import pandas as pd
import pkg_resources
from collections import defaultdict

from dateutil.relativedelta import relativedelta

class PipelineModelMetricCategoryYearSplitReport(AbstractReport):

    sub_folder = 'reports'
    scorer = MeanAbsoluteErrorScorer()
    date_column_name = None
    category_column_name = None
    category_list = []

    def report(self, pipeline: AbstractPipeline):
        # This report will run a instantiated scoring metric over the training and test set split over year intervals
        # as well as over specified categories with in category_list.

        report_list = []
        for i, cat in enumerate(self.category_list):

            report_dict = defaultdict(list)
            for model in pipeline.get_models( ):
                model_type = model.short_name
                report_dict['Model Type: '].append(model_type)

                # Train Set
                indices = pipeline.meta_data.index.values[
                    (pipeline.meta_data[self.category_column_name].isin(cat))
                    ]
                train = pipeline.train[pipeline.train.index.isin(indices)]
                train_y = pipeline.train_y[pipeline.train_y.index.isin(indices)]
                pred_y_train, _ = model.predict(train)
                report_dict['Training: ' + ' Size: ' + str(len(train_y))].append(
                    self.scorer.score(train_y, pred_y_train))

                # Test Set
                test_dates = pipeline.meta_data[self.date_column_name][
                    (pipeline.meta_data[self.category_column_name].isin(cat)) &
                    (pipeline.meta_data.index.isin(pipeline.test.index))
                    ]
                date_list = pd.to_datetime(test_dates.apply(lambda x: x.replace(day=1, month=1)).unique())
                for split_date in date_list:
                    indices = pipeline.meta_data.index.values[
                        (pipeline.meta_data[self.category_column_name].isin(cat)) &
                        (pipeline.meta_data[self.date_column_name] >= split_date) &
                        (pipeline.meta_data[self.date_column_name] < split_date + relativedelta(years=1))
                        ]
                    test = pipeline.test[pipeline.test.index.isin(indices)]
                    test_y = pipeline.test_y[pipeline.test_y.index.isin(indices)]
                    pred_y_test, _ = model.predict(test)
                    report_dict['Testing: ' + str(split_date.date()) + ' Size: '+ str(len(test_y))].append(
                        self.scorer.score(test_y, pred_y_test))

            report_df = pd.DataFrame(report_dict).T
            report_df['Category'] = i
            report_list.append(report_df)

        final_df = pd.concat(report_list)
        folder = get_cache_path()
        path = pkg_resources.resource_filename('crcdal', 'cache/' + folder + '/' + self.sub_folder +'/')
        pkg_resources.ensure_directory(path)
        final_df.to_csv(path + pipeline.dataset_tag + '_category_year_split_model_metrics_report.csv')
