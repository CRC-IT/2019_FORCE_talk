'''
Authored by Nathaniel Jones, 
Modified and maintained by the Big Data Analytics Team at California Resources Corporation
CC BY 2.0 License
'''

from crcdal.data_science_layer.pipeline.basic_regressor_pipeline import \
    BasicRegressorPipeline
from crcdal.data_science_layer.scoring.mean_absolute_error_scorer import \
    MeanAbsoluteErrorScorer
from crcdal.data_science_layer.machine_learning \
    .multi_layer_perceptrion_regressor import \
    MultiLayerPerceptronRegressorModel
from crcdal.data_science_layer.machine_learning.random_forest_regressor \
    import \
    RandomForestRegressorModel
from crcdal.data_science_layer.machine_learning.gradient_boost_regressor \
    import \
    GradientBoostRegressorModel
from crcdal.data_science_layer.machine_learning.k_neighbors_regressor import \
    KNeighborsRegressorModel
from crcdal.data_science_layer.machine_learning.linear_regression_regressor \
    import \
    LinearRegressionRegressorModel
from crcdal.data_science_layer.machine_learning.support_vector_regressor \
    import \
    SupportVectorRegressorModel
from crcdal.data_science_layer.machine_learning.dummy_regressor import \
    DummyRegressorModel
from crcdal.data_science_layer.preprocessing.default_scaler import \
    DefaultScaler
from crcdal.data_science_layer.preprocessing.default_normalizer import \
    DefaultNormalizer
from crcdal.data_science_layer.preprocessing.pca_decomp import PCADecomposition
from crcdal.data_science_layer.preprocessing.impute_missing import \
    ImputeMissing


class AddPayPipeline(BasicRegressorPipeline):
    def __init__(self):
        super().__init__()
        self.scorer = MeanAbsoluteErrorScorer()
        # self.set_parametric_scorer(scorer=self.scorer)
        self._feature_selection = None
        self.search_type = 'grid'

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
            if isinstance(ml, RandomForestRegressorModel):
                ml.hyper_param_dict = {'n_estimators': [
                    1, 2, 5, 10, 50, 100, 1000]}
            self.add_ml_model(ml=ml)

    def define_default_preprocessing(self):
        list_of_pre_processors = [
            DefaultScaler,
            ImputeMissing,
            DefaultNormalizer,
            # PCADecomposition,
        ]
        for processor in list_of_pre_processors:
            proc = processor()
            if isinstance(proc, PCADecomposition):
                proc.no_components = 3
        #     self.add_preprocessor(processor=proc)
        # y_processors = [DefaultScaler,
        #                 ]
        # for processor in y_processors:
        #     proc = processor()
        #     self.add_y_preprocessor(proc)

    # def _split_data(self, data, training_y):
    #     """Splitting by date"""
    #     test_index = int((1 - self.train_fraction) * len(data))
    #     self.test = data.iloc[:test_index]
    #     self.test_y = training_y.iloc[:test_index]
    #     self.train = data.iloc[test_index:]
    #     self.train_y = training_y.iloc[test_index:]
