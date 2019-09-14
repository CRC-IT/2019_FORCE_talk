from sklearn import svm
from sklearn.model_selection import RandomizedSearchCV


class RandomizedSearchCrossVal():
    estimator = svm.SVC(gamma='scale')
    param_distributions = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
    n_iter = 10
    scoring = None
    fit_params = None
    n_jobs = None
    iid = True
    refit = True
    cv = None
    verbose = 0
    pre_dispatch = '2 * n_jobs'
    random_state = 1
    error_score = 'raise -deprecating'
    return_train_score = 'warn'

    def search_models(self, x, y):
        RSCV = RandomizedSearchCV(
            estimator=self.estimator,
            param_distributions=self.param_distributions,
            n_iter=self.n_iter,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            iid=self.iid,
            refit=self.refit,
            cv=self.cv,
            verbose=self.verbose,
            pre_dispatch=self.pre_dispatch,
            random_state=self.random_state,
            error_score=self.error_score,
            return_train_score=self.return_train_score
        )
        RSCV.fit(x, y)

        return RSCV
