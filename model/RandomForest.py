__author__ = 'alexjzy'

from sklearn.ensemble import RandomForestRegressor


class RandomForest:
    def __init__(self, train_x, train_y):
        self.param = {
            'n_estimators': 100,
            'max_features': 'sqrt',
            'max_depth': 15,
            'random_state': 0
        }
        rf = RandomForestRegressor(**self.param)
        self.model = rf.fit(train_x, train_y)

    def predict(self, x):
        pred = self.model.predict(x)
        return pred


if __name__ == '__main__':
    pass
