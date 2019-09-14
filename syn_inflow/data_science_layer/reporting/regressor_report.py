from .abstract_report import AbstractReport
from ..pipeline.abstract_pipline import AbstractPipeline
import pkg_resources
import numpy as np
import pandas as pd
from ..utilities.cache_path import get_cache_path
from sklearn import metrics


class RegressorReport(AbstractReport):

    sub_folder = 'reports'

    def report(self, pipeline: AbstractPipeline):

        # Set Directory path
        folder = get_cache_path()
        path = pkg_resources.resource_filename('crcdal', 'cache/' + folder + '/' + self.sub_folder + '/')
        pkg_resources.ensure_directory(path)

        model_train_metrics = {}
        model_test_metrics = {}
        for i, model in enumerate(pipeline.get_models()):
            name = model.short_name
            preds_y_train, _ = model.predict(pipeline.train)
            preds_y_test, _ = model.predict(pipeline.test)

            preds_y_train = pd.DataFrame(preds_y_train)
            preds_y_test = pd.DataFrame(preds_y_test)
            train_y = pd.DataFrame(pipeline.train_y)
            test_y = pd.DataFrame(pipeline.test_y)

            # Account for multiple y values.
            k = 0
            for j in range(pipeline.test_y.shape[1]):
                model_train_metrics[str(name) + str(j)] = self._stats(train_y.iloc[:, j], preds_y_train.iloc[:, j])
                model_test_metrics[str(name) + str(j)] = self._stats(test_y.iloc[:, j], preds_y_test.iloc[:, j])
                k += 2

        # Feature Metrics
        df_model_train_metrics = pd.DataFrame(model_train_metrics)
        df_model_test_metrics = pd.DataFrame(model_test_metrics)
        df_train_test_count = pd.DataFrame({'train': [len(pipeline.train)], 'test': [len(pipeline.test)]})

        # Output to Excel
        try:
            writer = pd.ExcelWriter(path + pipeline.dataset_tag + '_reg_report.xlsx')
            df_model_train_metrics.to_excel(writer, 'Train Metrics')
            df_model_test_metrics.to_excel(writer, 'Test Metrics')
            df_train_test_count.to_excel(writer, 'Train Test Cnt')
            writer.save()
        except Exception as ex:
            print(ex)
            prepend_path = path + pipeline.dataset_tag
            df_model_train_metrics.to_csv(prepend_path + 'Train Metrics.csv')
            df_model_test_metrics.to_csv(prepend_path + 'Test Metrics.csv')
            df_train_test_count.to_csv(prepend_path + 'Train Test Cnt.csv')

    @staticmethod
    def _stats(true_y, pred_y):
        dic = {
            'EVS': metrics.explained_variance_score(true_y, pred_y),
            'MeanAE': metrics.mean_absolute_error(true_y, pred_y),
            'MeanSE': metrics.mean_squared_error(true_y, pred_y),
            'MSLE': metrics.mean_squared_log_error(np.abs(true_y), np.abs(pred_y)),
            'MedianAE': metrics.median_absolute_error(true_y, pred_y),
            'r2': metrics.r2_score(true_y, pred_y)
        }
        return dic