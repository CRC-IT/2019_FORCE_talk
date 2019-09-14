
from .abstract_pre_processor import \
    AbstractPreProcessor
from sklearn.decomposition.pca import PCA
import pandas as pd


class PCADecomposition(AbstractPreProcessor):
    pca = None
    no_components = 2

    def fit(self, data, y=None):
        self.pca = PCA(n_components=self.no_components)
        self.pca.fit(data)

    def fit_transform(self, data, y=None):
        self.fit(data, y)
        return self.transform(data, y)

    def transform(self, data, y=None):
        data = self._check_input(data)
        output = self.pca.transform(data)
        output = self._check_output(data, output)
        return output

    def _check_output(self, data, output):
        if isinstance(data, pd.DataFrame):
            columns = [
                'Component ' + str(x + 1) for x in range(self.no_components)]
            output = pd.DataFrame(data=output, columns=columns,
                                  index=data.index)
            return output
