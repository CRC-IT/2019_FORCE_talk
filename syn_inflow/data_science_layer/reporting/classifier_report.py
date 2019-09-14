from crcdal.data_science_layer.reporting.abstract_report import AbstractReport
from crcdal.data_science_layer.pipeline.abstract_pipline import AbstractPipeline
from crcdal.input_layer.configuration import Configuration
from ..utilities.cache_path import get_cache_path
from crcdal.data_layer.utilities.exception_tracking import ExceptionTracking
import pkg_resources
import pandas as pd
import numpy as np

from sklearn import metrics
import math

class ClassifierReport(AbstractReport):

    sub_folder = 'reports'

    def report(self, pipeline: AbstractPipeline):

        # Set Directory path
        subfolder = Configuration.get_cache_subfolder()
        path = pkg_resources.resource_filename('crcdal', 'cache/' + subfolder + '/reports/')
        pkg_resources.ensure_directory(path)

        # Results and Metric Dics
        model_test_metrics = {}
        model_train_metrics = {}

        for model in pipeline.get_models():
            name = model.short_name
            preds_y_test, probs_y_test = model.predict(pipeline.test)
            preds_y_train, probs_y_train = model.predict(pipeline.train)
            model_test_metrics.update({name: self._stats(pipeline.test_y, preds_y_test)})
            model_train_metrics.update({name: self._stats(pipeline.train_y, preds_y_train)})

        # Metrics at 50% threshold
        df_model_test_metrics = pd.DataFrame(model_test_metrics)
        df_model_train_metrics = pd.DataFrame(model_train_metrics)
        df_train_test_count = pd.DataFrame({'train': [len(pipeline.train)], 'test': [len(pipeline.test)]})

        # Output to Excel
        writer = pd.ExcelWriter(path + pipeline.dataset_tag + '_cls_report.xlsx')
        df_model_test_metrics.to_excel(writer, 'Test Metrics')
        df_model_train_metrics.to_excel(writer, 'Train Metrics')
        df_train_test_count.to_excel(writer, 'Train Test Cnt')
        writer.save()

    @staticmethod
    def _stats(true_y, pred_y):

        if pd.DataFrame(true_y).nunique()[0] > 2:
            average='weighted'
        else:
            average='binary'

        dic = {
            'accuracy': metrics.accuracy_score(true_y, pred_y),
            'recall': metrics.recall_score(true_y, pred_y, average=average),
            'precision': metrics.precision_score(true_y, pred_y, average=average),
            'f1': metrics.f1_score(true_y, pred_y, average=average),
            'kappa': metrics.cohen_kappa_score(true_y, pred_y),
            'confusionMatrix': metrics.confusion_matrix(true_y, pred_y),
            'MeanSE': metrics.mean_squared_error(true_y, pred_y),
            'RMeanSE': math.sqrt(metrics.mean_squared_error(true_y, pred_y)),
            'MeanAE': metrics.mean_absolute_error(true_y, pred_y),
            'MeanSLE': metrics.mean_squared_log_error(np.abs(true_y), pred_y),
            'R2': metrics.r2_score(true_y, pred_y),
            'MedianAE': metrics.median_absolute_error(true_y, pred_y),
            'EVS': metrics.explained_variance_score(true_y, pred_y),
            'CorrCoef': metrics.matthews_corrcoef(true_y,pred_y)
        }
        return dic