'''
Authored by Nathaniel Jones, 
Modified and maintained by the Big Data Analytics Team at California Resources Corporation
CC BY 2.0 License
'''

from ..pipeline.abstract_pipline import AbstractPipeline
from ..preprocessing.abstract_pre_processor import AbstractPreProcessor
from ..machine_learning.abstract_ml import AbstractML
import time
from sklearn.metrics.scorer import make_scorer
from ..reporting.models_and_metrics_deployment_report import \
    PipelineModelMetricDeployReport
from ..reporting.models_and_metrics_table import \
    PipelineModelMetricReports
from ..preprocessing.default_scaler import \
    DefaultScaler
from ..utilities.exception_tracking import ExceptionTracking
import pandas as pd


class BasicPipeline(AbstractPipeline):
    """Core implementation of a data science pipeline, should be extended
    via subclassing or runtime modification"""

    def __call__(self, data, y_result=None):
        """Called for deployment of a trained pipeline"""
        data = self._apply_deploy_preprocessing(data)
        data = self._apply_deploy_feature_selection(data)

        if self._best_model is None:
            raise (ValueError('Pipeline is not fit to any data'))
        else:
            short_name = self._best_model.short_name
            print('\nDeploying Best Model: ' + str(short_name))
        model_preds, model_probs = self._best_model.predict(data)
        output = model_preds
        output = self._invert_y_preprocessing(output)
        if y_result is not None:
            report = PipelineModelMetricDeployReport()
            report.report(self, data, y_result)
        return output

    def fit(self, data, training_y):
        """Fit a pipeline to training data"""
        start_time = time.time()
        start_i = start_time

        data, training_y = self.format_inputs(data, training_y)

        # Copy Original Data
        self.original_input_data = data.copy()
        self.original_y_data = training_y.copy()

        # Test/Train Split
        self._split_data(data, training_y)
        self._copy_original_split_data()

        # Preprocessing
        self._apply_training_preprocessing()
        # TODO Remove/Impute Empty/Bad Data and Features Here?

        # Sampling
        self._apply_sampling()

        # Feature Selection
        self._apply_feature_selection()

        # Train ML Models
        self._train_models(start_i)
        if self._best_model:
            print('\nBest Model: ' + str(self._best_model.short_name))
            print('Training Runtime: ' + str(time.time() - start_time))

        # Process Reports
        self._report_results()

    def format_inputs(self, data, training_y):
        return data, training_y

    def define_default_reports(self):
        list_of_reports = [
            PipelineModelMetricReports,
        ]
        for report in list_of_reports:
            rep = report()
            self.add_report(report=rep)

    def define_default_models(self):
        list_of_models = []
        for model in list_of_models:
            ml = model()
            self.add_ml_model(ml=ml)

    def define_default_feature_selector(self):
        self._feature_selection = None

    def define_default_preprocessing(self):
        list_of_pre_processors = [
            DefaultScaler
        ]
        for processor in list_of_pre_processors:
            proc = processor()
            self.add_preprocessor(processor=proc)

    def _report_results(self):
        for report in self._reports:
            if report is not None:
                try:
                    self._process_report(report)
                except:
                    continue

    def _apply_training_preprocessing(self):
        for processor in self._preprocessing:
            if processor is not None:
                self.train = self._fit_preprocessing(
                    processor, self.train, self.train_y)
                self.test = self._transform_preprocessing(
                    processor, self.test, self.test_y)
        for y_processor in self.preprocess_y:
            self.train_y = self._fit_preprocessing(
                y_processor, table_x=self.train_y)
            self.test_y = self._fit_preprocessing(
                y_processor, table_x=self.test_y)

    def _apply_deploy_preprocessing(self, data):
        for processor in self._preprocessing:
            if processor is not None:
                data = self._transform_preprocessing(processor,
                                                     data, y_data=None)
        return data

    def _train_models(self, start_i):
        failed_ml_model = []
        for idx, model in enumerate(self._ml_models):
            try:
                if model is not None:
                    print('\n->Training: ' + model.short_name)
                    model = self._apply_ml_model(model)
                    self._ml_models[idx] = model
                    delta_time = time.time() - start_i
                    print('Runtime: ' + str(delta_time))
                    start_i += delta_time
            except Exception as ex:
                print(ex)
                failed_ml_model.append(model)
                ExceptionTracking().log_exception('Data Science Model Failed', 'basic_pipeline', 'NA')
        [self._ml_models.remove(i) for i in failed_ml_model]

    def _apply_sampling(self):
        if self._sampler is not None:
            self._fit_sampler()

    def _apply_feature_selection(self):
        if self._feature_selection is not None:
            self._fit_feature_selection()

    def _apply_deploy_feature_selection(self, data):
        if self._feature_selection is not None:
            data = self._transform_feature_selection(data)
        return data

    def _copy_original_split_data(self):
        self.train_o = self.train.copy()
        self.test_o = self.test.copy()
        self.train_y_o = self.train_y.copy()
        self.test_y_o = self.test_y.copy()

    def _split_data(self, data, training_y):
        self.train, self.test, self.train_y, self.test_y = \
            self.splitter.train_test_split(
                table=data,
                y=training_y,
                train_size=self.train_fraction,
                random_state=self.random_seed)

    def _fit_preprocessing(
            self, processor: AbstractPreProcessor, table_x, table_y=None):
        output = processor.fit_transform(table_x, y=table_y)
        return output
        # self.test = processor.transform(self.test, y=self.test_y)

    def _transform_preprocessing(
            self, processor: AbstractPreProcessor, data, y_data=None):
        data_out = processor.transform(data, y=y_data)
        return data_out

    def _inverse_preprocessing(self, processor, data):
        try:
            return processor.inverse_transform(data)
        except Exception as ex:
            return data

    def _invert_y_preprocessing(self, predictions):
        for processor in reversed(self.preprocess_y):
            predictions = pd.DataFrame(predictions, columns = processor._input_data.columns)
            predictions = self._inverse_preprocessing(processor, predictions)
        return predictions

    def _fit_sampler(self):
        self.train, self.train_y = self._sampler.fit_sample(self.train, self.train_y)

    def _fit_feature_selection(self):
        self.train = self._feature_selection.select_features_from_model(self.train, self.train_y)
        self.test = self._feature_selection.select_features_in_test_set(self.test)
        print('features selected: ' + str(list(self._feature_selection.best_features)))

    def _transform_feature_selection(self, data):
        print('features selected: ' + str(list(self._feature_selection.best_features)))
        data = data[self._feature_selection.best_features]
        return data

    def _find_train_score(self, ml_model: AbstractML, **kwargs):
        if ml_model.model_search is not None:
            print('best_dev_set_cv_score: ' + str(ml_model.model_summary_results['best_score']))
            print('best_dev_set_cv_param: ' + str(ml_model.model_summary_results['best_param']))
        else:
            print('train_set_model_score: ' + str(ml_model.model_summary_results['best_score']))

    def _find_test_score(self, ml_model: AbstractML, **kwargs):
        # score best x-val model

        if self._parametric_scorer is not None:
            sc = self._parametric_scorer
            sc_name = 'parametric'
        else:
            sc = ml_model.scorer
            sc_name = 'model'

        custom_scorer = make_scorer(sc.score,
                                    greater_is_better=sc.greater_is_better,
                                    needs_proba=sc.needs_proba,
                                    needs_threshold=False,
                                    **kwargs)
        score = custom_scorer(ml_model.best_model, self.test, self.test_y, sample_weight=None)
        self.score_increasing= sc.greater_is_better
        self.models_results.append(score)
        print('test_set_' + str(sc_name) + '_score: ' + str(score))

    def _apply_ml_model(self, ml_model: AbstractML):
        # tell model to prep hyperparam search
        ml_model.search_models(x=self.train, y=self.train_y)

        # find train score
        kwargs = {}
        self._find_train_score(ml_model, **kwargs)

        # find test score
        kwargs = {}
        self._find_test_score(ml_model, **kwargs)

        # check if new best overall model
        if self._select_best_model():
            self._best_model = ml_model

        return ml_model

    def _select_best_model(self):

        new_score = self.models_results[len(self.models_results)-1]
        old_scores = self.models_results[:len(self.models_results)-1]
        for score in old_scores:
            if score is None or len(old_scores) == 0:
                return True
            if self.score_increasing and abs(score) > abs(new_score):
                return False
            if not self.score_increasing and abs(score) < abs(new_score):
                return False
        return True

    def _process_report(self, report):
        report.dataset_tag = self.dataset_tag
        report.report(self)
