__author__ = 'alexjzy'

import xgboost

class Xgboost:

    def __init__(self, train_x, train_y):
        self.param = {
            'n_estimators': 100,
            'learning_rate': 0.09,
            'gamma': 0.1,
            'subsample': 0.51,
            'colsample_bytree': 0.9,
            'max_depth': 8,
            'alpha': 1,
            'reg_lambda': 9
        }

        xgbModel = xgboost.XGBRegressor(**self.param)
        self.model = xgbModel.fit(train_x, train_y)


    def predict(self, x):
        y = self.model.predict(x)
        return y

if __name__ == '__main__':
    pass
