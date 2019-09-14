from .abstract_report import AbstractReport
from ..pipeline.abstract_pipline import AbstractPipeline
from ..utilities.exception_tracking import ExceptionTracking
import pkg_resources
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ..utilities.cache_path import get_cache_path


class RegressorCurves(AbstractReport):

    sub_folder = 'reports'
    log_y = False
    exp_y = False

    def report(self, pipeline: AbstractPipeline):

        # Set Directory path
        folder = get_cache_path()
        path = pkg_resources.resource_filename('crcdal', 'cache/' + folder + '/' + self.sub_folder + '/')
        pkg_resources.ensure_directory(path)

        # Hist Train
        fig, ax = plt.subplots(figsize=(40, 40))
        pipeline.train.hist(bins=100, ax=ax)
        fig.savefig(path + 'Hist_Train.png')

        # Hist Test
        fig, ax = plt.subplots(figsize=(40, 40))
        pipeline.test.hist(bins=100, ax=ax)
        fig.savefig(path + 'Hist_Test.png')

        # Feature Results
        nrows = len(pipeline._ml_models)
        nrows = 2 if nrows == 1 else nrows
        ncols = 2
        ncols = 2 ** pipeline.test_y.shape[1] if pipeline.test_y.shape[1] > 1 else ncols

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=False, sharey=False, figsize=(40, 10 * nrows))
        fig2, axes2 = plt.subplots(nrows=nrows, ncols=ncols, sharex=False, sharey=False, figsize=(40, 10 * nrows))
        for i, model in enumerate(pipeline.get_models()):
            name = model.short_name
            preds_y_train, _ = model.predict(pipeline.train)
            preds_y_test, _ = model.predict(pipeline.test)

            preds_y_train = pd.DataFrame(preds_y_train)
            preds_y_test = pd.DataFrame(preds_y_test)
            train_y = pd.DataFrame(pipeline.train_y)
            test_y = pd.DataFrame(pipeline.test_y)

            k = 0
            for j in range(pipeline.test_y.shape[1]):

                try:
                    sns.distplot(preds_y_test.iloc[:, j], label='predict', ax=axes[i, k])
                    sns.distplot(test_y.iloc[:, j], label='test', ax=axes[i, k])
                    axes[i, k].set_title('Distribution ' + str(name))
                    axes[i, k].legend()

                    sns.regplot(test_y.iloc[:, j], preds_y_test.iloc[:, j], ax=axes[i, k + 1])
                    axes[i, k + 1].set_title('Scatter ' + str(name))
                    axes[i, k + 1].set_xlabel('Test')
                    axes[i, k + 1].set_ylabel('Predict')

                    sns.distplot(np.exp(preds_y_test.iloc[:, j]), label='predict', ax=axes2[i, k])
                    sns.distplot(np.exp(test_y.iloc[:, j]), label='test', ax=axes2[i, k])
                    axes2[i, k].set_title('Distribution ' + str(name))
                    axes2[i, k].legend()

                    sns.regplot(np.exp(test_y.iloc[:, j]), np.exp(preds_y_test.iloc[:, j]), ax=axes2[i, k + 1])
                    axes2[i, k + 1].set_title('Scatter ' + str(name))
                    axes2[i, k + 1].set_xlabel('Test')
                    axes2[i, k + 1].set_ylabel('Predict')
                except:
                    ExceptionTracking().log_exception('Result distributions plot failed', 'DCE_pipeline', 'NA')

                k += 2

        fig.savefig(path + 'result_distributions_log.png')
        fig2.savefig(path + 'result_distributions.png')
