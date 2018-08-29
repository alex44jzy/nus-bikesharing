__author__ = 'alexjzy'

from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
import numpy as np


class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    def fit(self, X, y, features):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=False)

        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            x_model = X[features[i]]
            for train_index, holdout_index in kfold.split(x_model, y):
                instance = model
                # instance.fit(X[train_index], y[train_index])
                instance.fit(x_model.iloc[train_index], y[train_index])
                self.base_models_[i].append(instance)

                if i == 0:
                    y_pred = instance.predict(x_model.iloc[holdout_index]).flatten()
                else:
                    y_pred = instance.predict(x_model.iloc[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred

        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    def predict(self, x, features):
        small_model_container = [list() for x in self.base_models]
        for i, base_model in enumerate(self.base_models_):  # 3
            print(base_model)
            small_model_container[i] = []
            for small_base_model in base_model:  # 5
                pred = small_base_model.predict(x[features[i]])
                small_model_container[i].append(pred)
            small_model_container[i] = np.column_stack([lst for lst in small_model_container[i]]).mean(axis=1)
        res = np.transpose(np.asarray(small_model_container))
        return self.meta_model_.predict(res)



