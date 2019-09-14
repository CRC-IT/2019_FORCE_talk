from .abstract_report import AbstractReport
from ..pipeline.abstract_pipline import AbstractPipeline
import pkg_resources
import seaborn as sns
import pandas as pd
from ..utilities.cache_path import get_cache_path


class PairPlot(AbstractReport):

    sub_folder = 'reports'
    data_set = 'all_train'
    sample_size = None

    def report(self, pipeline: AbstractPipeline):

        # Set Directory path
        folder = get_cache_path()
        path = pkg_resources.resource_filename('crcdal', 'cache/' + folder + '/' + self.sub_folder +'/')
        pkg_resources.ensure_directory(path)

        # Select Data Set
        if self.data_set == 'all_train':
            d = pd.merge(left=pd.DataFrame(pipeline.train), right=pd.DataFrame(pipeline.train_y)
                         , how='inner', left_index=True, right_index=True)
        elif self.data_set == 'train':
            d = pipeline.train
        elif self.data_set == 'train_y':
            d = pipeline.train_y
        elif self.data_set == 'all_test':
            d = pd.merge(left=pd.DataFrame(pipeline.test), right=pd.DataFrame(pipeline.test_y)
                         , how='inner', left_index=True, right_index=True)
        elif self.data_set == 'test':
            d = pipeline.test
        elif self.data_set == 'test_y':
            d = pipeline.test_y
        else:
            d = pipeline.train

        # Pairplot
        if self.sample_size is not None:
            sns.pairplot(d.sample(self.sample_size)).savefig(path + 'pairplot.png')
        else:
            sns.pairplot(d).savefig(path + 'pairplot.png')