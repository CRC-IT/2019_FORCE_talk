'''
Authored by Nathaniel Jones, 
Modified and maintained by the Big Data Analytics Team at California Resources Corporation
CC BY 2.0 License
'''

from abc import ABC, abstractmethod
from ..preprocessing.abstract_pre_processor import AbstractPreProcessor
from ..machine_learning.abstract_ml import AbstractML
from ..reporting.abstract_report import AbstractReport
from ..data_split.split_with_index import SplitWithIndex
from ..scoring.abstract_scorer import AbstractScorer
from ..scoring.r2_scorer import R2Scorer

class AbstractPipeline(ABC):
    """Definition of the functionality of a data science pipeline"""
    def __init__(self):
        super().__init__()
        self._scorer = R2Scorer()
        self._search_type = None
        self._random_seed = 1
        self.original_input_data = None
        self.original_y_data = None
        self.test = None
        self.train = None
        self.train_y = None
        self.test_y = None
        self._preprocessing = []
        self._feature_selection = None
        self._ml_models = []
        self._sampler = None
        self._reports = []
        self._scorers = []
        self._parametric_scorer = None
        self._best_model = None
        self._cross_validator = None
        self.meta_data = None
        self.input_x = None
        self.input_y = None
        self.unlabelled_x = None
        self.splitter = SplitWithIndex()
        self.train_fraction = 0.8
        self.models_results = []
        self.dataset_tag = ''
        self.define_default_preprocessing()
        self.define_default_models()
        self.define_default_reports()
        self.define_default_feature_selector()
        self.preprocess_y = []

        # self.scorer = R2Scorer()


    @property
    def scorer(self) -> AbstractScorer:
        if self._scorer is None:
            self._scorer = R2Scorer()
        return self._scorer

    @scorer.setter
    def scorer(self, scorer: AbstractScorer):
        self._scorer = scorer
        for model in self._ml_models:
            model.scorer = scorer
            model.search_scoring = scorer

    @property
    def random_seed(self):
        return self._random_seed

    @random_seed.setter
    def random_seed(self, value):
        self._random_seed = value
        for model in self._ml_models:
            model.random_state = value
            model.search_random_state = value

    @property
    def search_type(self):
        return self._search_type

    @search_type.setter
    def search_type(self, value: str):
        self._search_type = value
        for model in self._ml_models:
            model.model_search = value

    @abstractmethod
    def __call__(self, data):
        raise (NotImplementedError('Do not use ABC method'))

    @abstractmethod
    def fit(self, data, training_y):
        raise (NotImplementedError('Do not use ABC method'))

    def add_preprocessor(self, *, processor: AbstractPreProcessor):
        self._preprocessing.append(processor)

    def add_y_preprocessor(self, processor: AbstractPreProcessor):
        self.preprocess_y.append(processor)

    def add_feature_selector(self, *, feature_selector):
        self._feature_selection = feature_selector

    def add_sampler(self, *, sampler):
        self._sampler = sampler

    def add_ml_model(self, *, ml: AbstractML):
        """Adds a model to the pipeline, will update the seed, scorer, and
        hyper param search to the pipeline's overall settings"""
        ml.scorer = self.scorer
        ml.search_random_state = self.random_seed
        ml.random_state = self.random_seed
        ml.model_search = self.search_type
        self._ml_models.append(ml)

    def add_report(self, *, report: AbstractReport):
        self._reports.append(report)

    def add_scorer(self, *, scorer):
        self._scorers.append(scorer)

    def set_parametric_scorer(self, *, scorer):
        self._parametric_scorer = scorer

    def set_cross_validator(self, *, cross_validator):
        self._cross_validator = cross_validator

    def get_models(self):
        for model in self._ml_models:
            yield model

    def define_default_models(self):
        pass

    def define_default_preprocessing(self):
        pass

    def define_default_reports(self):
        pass

    def define_default_feature_selector(self):
        pass

    def is_fit(self):
        if self._best_model is None:
            return False
        return True
