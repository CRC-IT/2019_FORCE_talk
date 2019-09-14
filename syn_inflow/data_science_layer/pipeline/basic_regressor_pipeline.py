'''
Authored by Nathaniel Jones,
Modified and maintained by the Big Data Analytics Team at California Resources Corporation
CC BY 2.0 License
'''

from ..pipeline.basic_pipeline import BasicPipeline
from ..feature_selection.extra_trees_regressor_feature_selection import ExtraTreesRegressorFeatureSelector
from ..preprocessing.default_scaler import DefaultScaler
from ..preprocessing.default_normalizer import DefaultNormalizer
from ..machine_learning.multi_layer_perceptrion_regressor import MultiLayerPerceptronRegressorModel
from ..machine_learning.random_forest_regressor import RandomForestRegressorModel
from ..machine_learning.gradient_boost_regressor import GradientBoostRegressorModel
from ..machine_learning.k_neighbors_regressor import KNeighborsRegressorModel
from ..machine_learning.linear_regression_regressor import LinearRegressionRegressorModel
from ..machine_learning.support_vector_regressor import SupportVectorRegressorModel
from ..machine_learning.dummy_regressor import DummyRegressorModel
from ..scoring.r2_scorer import R2Scorer
from ..reporting.models_and_metrics_table import PipelineModelMetricReports
from ..reporting.extra_trees_regressor_feature_weights import ExtraTreesRegressorFeatureWeightsReport
from ..reporting.pair_plot import PairPlot
from ..reporting.regressor_report import RegressorReport
from ..reporting.regressor_curves import RegressorCurves
from ..reporting.feature_importance_report import FeatureImportanceReport

class BasicRegressorPipeline(BasicPipeline):
    """Core implementation of a data science pipeline,
    should be extended via subclassing or runtime modification"""

    def __init__(self):
        super().__init__()

        # us = UpSampler()
        # us.random_state = self.random_seed
        # self.add_sampler(sampler=us)
        # ds = DownSampler()
        # ds.random_state = self.random_seed
        # self.add_sampler(sampler=ds)

        # Scorer
        self.scorer = R2Scorer()
        # self.set_parametric_scorer(scorer=self.scorer)
        self.dataset_tag = 'basic_reg_pipeline'
        # self.search_type = 'grid'

    def define_default_models(self):
        list_of_models = [
            MultiLayerPerceptronRegressorModel,
            RandomForestRegressorModel,
            GradientBoostRegressorModel,
            KNeighborsRegressorModel,
            LinearRegressionRegressorModel,
            SupportVectorRegressorModel,
            DummyRegressorModel
        ]
        for model in list_of_models:
            ml = model()
            self.add_ml_model(ml=ml)

    def define_default_preprocessing(self):
        list_of_pre_processors = [
            DefaultScaler,
            DefaultNormalizer
        ]
        for processor in list_of_pre_processors:
            proc = processor()
            self.add_preprocessor(processor=proc)

    def define_default_reports(self):
        list_of_reports = [
            PipelineModelMetricReports,
            ExtraTreesRegressorFeatureWeightsReport,
            PairPlot,
            RegressorReport,
            RegressorCurves,
            FeatureImportanceReport
        ]
        for report in list_of_reports:
            rep = report()
            self.add_report(report=rep)

    def define_default_feature_selector(self):
        etfs = ExtraTreesRegressorFeatureSelector()
        etfs.random_state = self.random_seed
        self.add_feature_selector(feature_selector=etfs)
