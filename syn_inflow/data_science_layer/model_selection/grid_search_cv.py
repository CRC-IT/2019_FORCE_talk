from sklearn import svm
from sklearn.model_selection import GridSearchCV


class GridSearchCrossVal():
    estimator = svm.SVC(gamma='scale')
    param_grid = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
    scoring = None
    fit_params = None
    n_jobs = None
    iid = True
    refit = True
    cv = None
    verbose = 0
    pre_dispatch = '2 * n_jobs'
    error_score = 'raise'
    return_train_score = 'warn'

    def search_models(self, x, y):
        GSCV = GridSearchCV(
            estimator=self.estimator,
            param_grid=self.param_grid,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            iid=self.iid,
            refit=self.refit,
            cv=self.cv,
            verbose=self.verbose,
            pre_dispatch=self.pre_dispatch,
            error_score=self.error_score,
            return_train_score=self.return_train_score
        )
        GSCV.fit(x, y, self.fit_params)

        return GSCV
