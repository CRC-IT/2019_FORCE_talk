from .abstract_report import AbstractReport
from ..pipeline.abstract_pipline import AbstractPipeline
from ..utilities.cache_path import get_cache_path
import pkg_resources
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics

class ClassifierCurves(AbstractReport):

    sub_folder = 'reports'

    def report(self, pipeline: AbstractPipeline):

        # Set Directory path
        subfolder = get_cache_path()
        path = pkg_resources.resource_filename('crcdal', 'cache/' + subfolder + '/reports/')
        pkg_resources.ensure_directory(path)

        # Hist Train
        fig, ax = plt.subplots(figsize=(40, 40))
        pipeline.train.hist(bins=100, ax=ax)
        fig.savefig(path + 'Hist_Train.png')

        # Hist Test
        fig, ax = plt.subplots(figsize=(40, 40))
        pipeline.test.hist(bins=100, ax=ax)
        fig.savefig(path + 'Hist_Test.png')

        # Results and Metric Dics
        fpr = {}
        tpr = {}
        roc_t = {}
        fpr_train = {}
        tpr_train = {}
        roc_t_train = {}
        p = {}
        r = {}
        pr_t = {}
        auc = {}
        auc_train = {}

        for model in pipeline.get_models():
            name = model.short_name
            preds_y_test, probs_y_test = model.predict(pipeline.test)
            preds_y_train, probs_y_train = model.predict(pipeline.train)

            try:
                fp_rate, tp_rate, roc_threshold = metrics.roc_curve(pipeline.test_y, probs_y_test[:, 1])
            except:
                fp_rate, tp_rate, roc_threshold = [0,0,0]
            fpr.update({name: fp_rate})
            tpr.update({name: tp_rate})
            roc_t.update({name: roc_threshold})

            try:
                fp_rate_train, tp_rate_train, roc_threshold_train = metrics.roc_curve(pipeline.train_y, probs_y_train[:, 1])
            except:
                fp_rate_train, tp_rate_train, roc_threshold_train = [0,0,0]
            fpr_train.update({name: fp_rate_train})
            tpr_train.update({name: tp_rate_train})
            roc_t_train.update({name: roc_threshold_train})

            try:
                precision, recall, pr_thresholds = metrics.precision_recall_curve(pipeline.test_y, probs_y_test[:, 1])
            except:
                precision, recall, pr_thresholds = [0,0,0]
            p.update({name: precision})
            r.update({name: recall})
            pr_t.update({name: pr_thresholds})

            try:
                au_curve = metrics.roc_auc_score(pipeline.test_y, probs_y_test[:, 1])
            except:
                au_curve = 0
            auc.update({name: au_curve})

            try:
                au_curve_train = metrics.roc_auc_score(pipeline.train_y, probs_y_train[:, 1])
            except:
                au_curve_train = 0
            auc_train.update({name: au_curve_train})

        # ROC Curves - Test Set
        plt.figure(None, figsize=(10, 10))
        plt.plot([0, 1], [0, 1], 'k--')
        for model in pipeline.get_models():
            name = model.short_name
            plt.plot(fpr[name], tpr[name], label=str(name) + ': ' + str(auc[name]))
        plt.xlim(0, 1)
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
        plt.savefig(path + 'ROC_Curves_Test.png')

        # ROC Curves - Train Set
        plt.figure(None, figsize=(10, 10))
        plt.plot([0, 1], [0, 1], 'k--')
        for model in pipeline.get_models():
            name = model.short_name
            plt.plot(fpr_train[name], tpr_train[name], label=str(name) + ': ' + str(auc_train[name]))
        plt.xlim(0, 1)
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
        plt.savefig(path + 'ROC_Curves_Train.png')

        # Precision Recall Curves
        plt.figure(None, figsize=(10, 10))
        for model in pipeline.get_models():
            name = model.short_name
            plt.plot(r[name], p[name], label=str(name))
        plt.xlabel('Recall rate')
        plt.ylabel('Precision rate')
        plt.legend(loc='best')
        plt.xlim([0, 1])
        plt.savefig(path + 'PR_Curves.png')

        # P&R Threshold Curves
        fig, axes = plt.subplots(nrows=1, ncols=2, sharex='all', sharey='all', figsize=(20, 10))
        for model in pipeline.get_models():
            name = model.short_name
            pr_t2 = pr_t[name]
            if isinstance(pr_t[name],int):
                pr_t2 = [pr_t2]
            if len(pr_t2) > 1:
                pr_t2 = np.append(pr_t2, 1)
            # Precision
            axes[0].plot(pr_t2, p[name], label=str(name))
            # Recall
            axes[1].plot(pr_t2, r[name], label=str(name))
        axes[0].set_xlabel('Threshold')
        axes[0].set_ylabel('Precision rate')
        axes[1].set_xlabel('Threshold')
        axes[1].set_ylabel('Recall rate')
        axes[0].legend(loc='best')
        axes[1].legend(loc='best')
        axes[0].set_xlim([0, 1])
        axes[1].set_xlim([0, 1])
        plt.savefig(path + 'PR_Threshold_Curves.png')

        # True & False Rate Threshold Curves
        fig, axes = plt.subplots(nrows=1, ncols=2, sharex='all', sharey='all', figsize=(20, 10))
        for model in pipeline.get_models():
            name = model.short_name
            # True Positve
            axes[0].plot(roc_t[name], tpr[name], label=str(name))
            # False Positve
            axes[1].plot(roc_t[name], fpr[name], label=str(name))
        axes[0].set_xlabel('Threshold')
        axes[0].set_ylabel('True positive rate')
        axes[1].set_xlabel('Threshold')
        axes[1].set_ylabel('False positive rate')
        axes[0].legend(loc='best')
        axes[1].legend(loc='best')
        axes[0].set_xlim([0, 1])
        axes[1].set_xlim([0, 1])
        plt.savefig(path + 'TF_Threshold_Curves.png')
