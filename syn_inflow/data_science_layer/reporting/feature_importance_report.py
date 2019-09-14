from .abstract_report import AbstractReport
from ..pipeline.abstract_pipline import AbstractPipeline
from ..machine_learning.random_forest_classifier import RandomForestClassifierModel
from ..machine_learning.gradient_boost_classifier import GradientBoostClassifierModel
from ..machine_learning.extra_trees_classifier import ExtraTreesClassifierModel
from ..machine_learning.random_forest_regressor import RandomForestRegressorModel
from ..machine_learning.gradient_boost_regressor import GradientBoostRegressorModel
from ..machine_learning.extra_trees_regressor import ExtraTreesRegressorModel
from ..utilities.cache_path import get_cache_path
import pandas as pd

import pkg_resources


class FeatureImportanceReport(AbstractReport):

    sub_folder = 'reports'

    def report(self, pipeline: AbstractPipeline):

        model_list = (RandomForestClassifierModel, GradientBoostClassifierModel, ExtraTreesClassifierModel,
                      RandomForestRegressorModel, GradientBoostRegressorModel, ExtraTreesRegressorModel)

        report_list = []
        for model in pipeline.get_models( ):
            if isinstance(model, model_list):
                ft = pd.DataFrame([pipeline.train.columns.values, model.best_model.feature_importances_],
                              index=['Feature', 'Weight']).transpose().sort_values(by=['Weight'], ascending=False)
                ft['Model'] = model.short_name
                report_list.append(ft)

        report_df = pd.concat(report_list)
        folder = get_cache_path()
        path = pkg_resources.resource_filename('crcdal', 'cache/' + folder + '/' + self.sub_folder +'/')
        pkg_resources.ensure_directory(path)
        report_df.to_csv(path + pipeline.dataset_tag + '_model_feature_importance_report.csv')

