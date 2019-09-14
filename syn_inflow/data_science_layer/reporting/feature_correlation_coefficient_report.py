from .abstract_report import AbstractReport
from ..pipeline.abstract_pipline import AbstractPipeline
import pkg_resources
from ..utilities.cache_path import get_cache_path
from yellowbrick.target import FeatureCorrelation


class FeatureCorrelationCoefficientReport(AbstractReport):

    sub_folder = 'reports'

    def report(self, pipeline: AbstractPipeline):

        folder = get_cache_path()
        path = pkg_resources.resource_filename('crcdal', 'cache/' + folder + '/' + self.sub_folder +'/')
        pkg_resources.ensure_directory(path)

        feature_names = list(pipeline.train.columns())
        visualizer = FeatureCorrelation(labels=feature_names)
        visualizer.fit(pipeline.train, pipeline.train_y)
        visualizer.poof(outpath=path + pipeline.dataset_tag + '_model_feature_correlation_report.csv')



