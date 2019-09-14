from .abstract_report import AbstractReport
from ..pipeline.abstract_pipline import AbstractPipeline
import pandas as pd
import matplotlib.pyplot as plt
import pkg_resources
from ..utilities.cache_path import get_cache_path


class ActualVsPredictionPlot(AbstractReport):

    sub_folder = 'reports'

    def report(self, pipeline: AbstractPipeline):
        x = pipeline.train
        y = pipeline.train_y
        pred_y = pipeline(x)
        plt.scatter(y, pred_y)
        plt.suptitle('Predicted vs Actual', fontsize=18, y=1.0)
        plt.xlabel('Actual', fontsize=22)
        plt.ylabel('Predicted', fontsize=22)
        plt.legend( )
        folder = get_cache_path()
        path = pkg_resources.resource_filename('crcdal', 'cache/' + folder + '/' + self.sub_folder +'/')
        pkg_resources.ensure_directory(path)
        plt.savefig(path + self.dataset_tag + '_Predicted_vs_actual.jpg')
