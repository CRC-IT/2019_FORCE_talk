from .abstract_report import AbstractReport
from ..pipeline.abstract_pipline import AbstractPipeline
import pandas as pd
import pkg_resources
from collections import defaultdict
from ..utilities.cache_path import get_cache_path


class PipelineModelMetricReports(AbstractReport):

    sub_folder = 'reports'

    def report(self, pipeline: AbstractPipeline):
        report_dict = defaultdict(list)
        for model in pipeline.get_models( ):
            model_type = model.short_name
            report_dict['model_type'].append(model_type)
            pred_y_train, _ = model.predict(pipeline.train)
            pred_y_test, _ = model.predict(pipeline.test)

            true_y = pipeline.train_y
            pred_y = pred_y_train
            dic_train = {
                'Training Set Mean Squared Log Error': 'mean_squared_log_error(np.abs(true_y), np.abs(pred_y))',
                'Training Set Root Mean Squared Error': 'math.sqrt(mean_squared_error(true_y, pred_y))',
                'Training Set Mean Absolute Error': 'mean_absolute_error(true_y, pred_y)',
                'Training Set R2': 'r2_score(true_y, pred_y)',
                'Training Set Median Absolute Error': 'median_absolute_error(true_y, pred_y)',
                'Training Set Explained Variance Score': 'explained_variance_score(true_y, pred_y)',
            }
            for k, v in dic_train.items():
                try:
                    report_dict[k].append(eval(v))
                except:
                    del report_dict[k]

            true_y = pipeline.test_y
            pred_y = pred_y_test
            dic_test = {
                'Test Set Mean Squared Log Error': 'mean_squared_log_error(np.abs(true_y), np.abs(pred_y))',
                'Test Set Root Mean Squared Error': 'math.sqrt(mean_squared_error(true_y, pred_y))',
                'Test Set Mean Absolute Error': 'mean_absolute_error(true_y, pred_y)',
                'Test Set R2': 'r2_score(true_y, pred_y)',
                'Test Set Median Absolute Error': 'median_absolute_error(true_y, pred_y)',
                'Test Set Explained Variance Score': 'explained_variance_score(true_y, pred_y)'
            }
            for k, v in dic_test.items():
                try:
                    report_dict[k].append(eval(v))
                except:
                    del report_dict[k]

        report_df = pd.DataFrame(report_dict).T
        folder = get_cache_path()
        path = pkg_resources.resource_filename('crcdal', 'cache/' + folder + '/' + self.sub_folder +'/')
        pkg_resources.ensure_directory(path)
        report_df.to_csv(path + pipeline.dataset_tag + '_model_metrics_report.csv')

