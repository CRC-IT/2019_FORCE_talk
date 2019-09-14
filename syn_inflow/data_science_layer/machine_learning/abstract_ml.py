
from abc import ABC, abstractmethod
from ..model_selection.grid_search_cv import GridSearchCrossVal
from ..model_selection.random_search_cv import RandomizedSearchCrossVal
from collections import defaultdict
from sklearn.metrics.scorer import make_scorer
from crcdal.data_layer.utilities.exception_tracking import ExceptionTracking


class AbstractML(ABC):
    """General wrapper for all machine learning algorithms"""
    short_name = None
    model_summary_results = {}
    model_search = None
    hyper_param_dict = None
    scorer = None
    current_model = None
    best_model = None
    upper_threshold = None
    lower_threshold = None
    sklearn_model = None
    default_model = None
    models = []
    search_scoring = None
    search_cv = None
    search_n_iter = None
    search_n_jobs = None
    search_verbose = None
    search_random_state = None

    def __init__(self):
        self.model_summary_results = {}
        self.set_params()

    def set_params(self):
        if self.sklearn_model is not None:
            params = self.sklearn_model.get_params()
            for k, v in params.items():
                setattr(self, k, v)

    def set_default_model(self):
        if self.sklearn_model is not None:
            param_names = self.sklearn_model._get_param_names()
            params = {p: getattr(self, p) for p in param_names}
            self.default_model = self.sklearn_model.set_params(**params)
        else:
            self.default_model = None

    def search_models(self, x, y, **kwargs):
        self.set_default_model()
        self.cv_search_models(x, y, **kwargs)

    def cv_search_models(self, x, y, **kwargs):
        model_results = defaultdict(list)

        if self.check_should_run_search():
            best_param, best_score, current_score = self.search_current_model(
                x, y, kwargs)
        else:
            best_param, best_score, current_score = self.fit_model(
                x, y, kwargs)
        model_results['models'] = self.current_model
        model_results['scores'] = current_score
        model_results['params'] = self.hyper_param_dict
        model_results['best_score'] = best_score
        model_results['best_param'] = best_param
        self.models = model_results['models']
        self.model_summary_results['scores'] = model_results['scores']
        self.model_summary_results['params'] = model_results['params']
        self.model_summary_results['best_score'] = model_results['best_score']
        self.model_summary_results['best_param'] = model_results['best_param']

    def fit_model(self, x, y, kwargs):
        self.current_model = self.default_model
        self._fit(x, y)
        self.best_model = self.current_model
        return self.score_model(x, y, kwargs)

    def search_current_model(self, x, y, kwargs):
        scorer = self.make_scorer_for_search(kwargs)
        if self.model_search.lower() == 'grid':
            cf = self.build_search(scorer)
        else:
            cf = self.build_search(scorer, RandomizedSearchCrossVal)
        self.current_model = cf.search_models(x, y)
        self.best_model = self.current_model.best_estimator_
        current_score = self.current_model.cv_results_['mean_test_score']
        best_param = self.current_model.best_params_
        best_score = self.current_model.best_score_
        return best_param, best_score, current_score

    def score_model(self, x, y, kwargs):
        scorer = self.make_scorer_for_search(kwargs)
        current_score = None
        try:
            current_score = scorer(self.current_model, x, y,
                                   sample_weight=None)
        except Exception as ex:
            ExceptionTracking().log_exception("test", 'test', 'test')
        best_param = None
        best_score = current_score
        return best_param, best_score, current_score

    def fit(self, x, y):
        if self.current_model is None:
            self.search_models(x, y)
        elif self.best_model is None:
            self.search_models(x, y)

    @abstractmethod
    def predict(self, x):
        raise (NotImplementedError('Do not use ABC class method'))

    def _fit(self, x, y):
        self.current_model.fit(x, y)

    def _score(self, true_y, predicted_y, **kwargs):
        return self.scorer.score(true_y, predicted_y, **kwargs)

    def build_search(self, custom_scorer, class_func=GridSearchCrossVal):
        cf = class_func()
        cf.estimator = self.default_model
        cf.cv = self.search_cv
        cf.scoring = custom_scorer
        cf.n_jobs = self.search_n_jobs
        cf.verbose = self.search_verbose
        if isinstance(cf, RandomizedSearchCrossVal):
            cf.param_distributions = self.hyper_param_dict
            cf.n_iter = self.search_n_iter
            cf.random_state = self.search_random_state
        else:
            cf.param_grid = self.hyper_param_dict
        return cf

    def check_should_run_search(self):
        check1 = False
        if self.model_search is not None:
            check1 = self.model_search.lower() in ['grid', 'random']
        check2 = self.hyper_param_dict is not None
        return check1 and check2

    def make_scorer_for_search(self, kwargs):
        if self.check_should_run_search():
            custom_scorer = make_scorer(self.search_scoring.score,
                                        greater_is_better=self.search_scoring.greater_is_better,
                                        needs_proba=self.search_scoring.needs_proba,
                                        needs_threshold=False,
                                        **kwargs)
        else:
            custom_scorer = make_scorer(self.scorer.score,
                                        greater_is_better=self.scorer.greater_is_better,
                                        needs_proba=self.scorer.needs_proba,
                                        needs_threshold=False,
                                        **kwargs)
        return custom_scorer
