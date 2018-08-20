__author__ = 'alexjzy'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

class Preprocessing():
    def __init__(self):
        self.filename = '/day.csv'

    def load_data(self):
        print('Start loading dataset')
        data_dir = '/dataset'
        os.chdir('./..')
        cwd = os.getcwd() + data_dir
        print(cwd)
        data_file_name = self.filename
        dataset = pd.read_csv(cwd + data_file_name)
        return dataset

    def calcAvgShift(self, dfCol, period, exclude):
        try:
            summary = dfCol.rolling(period).mean()
            if period is None:
                return dfCol
            if exclude is None:
                return summary
            mean = dfCol.shift(exclude).rolling(period, 1).mean()
            return mean.apply(lambda x: round(x, 3))
        except Exception as e:
            print("error:" + str(e))

    def calcIncreaseRatio(self, pre, comp):
        inc = (comp - pre) / pre
        return round(inc, 3)

    def verboseClean(self, dataset):
        # convert the data time type
        dataset.dteday = dataset.dteday.apply(lambda x: pd.to_datetime(x))
        dataset = dataset.drop(['instant'], axis=1)
        dataset = dataset.rename(columns={'registered': 'regist'})
        return dataset

    def featureEngineering(self, dataset):
        dataset['casual_lag2'] = dataset['casual'].shift(2)
        dataset['regist_lag2'] = dataset['regist'].shift(2)

        dataset['temp_inc'] = self.calcIncreaseRatio(dataset.temp.shift(1), dataset.temp)
        dataset['temp_inc2day'] = self.calcIncreaseRatio(dataset.temp.shift(2), dataset.temp.shift(1))
        dataset['hum_inc2day'] = self.calcIncreaseRatio(dataset.hum.shift(2), dataset.hum.shift(1))

        dataset['casual_avg_aheadWeek'] = self.calcAvgShift(dataset.casual, 7, 2)
        dataset['regist_avg_aheadWeek'] = self.calcAvgShift(dataset.regist, 7, 2)

        dataset['casual_avg_ahead3days'] = self.calcAvgShift(dataset.casual, 5, 2)
        dataset['regist_avg_ahead3days'] = self.calcAvgShift(dataset.regist, 5, 2)

        dataset['casual_avg_aheadMonth'] = self.calcAvgShift(dataset.casual, 31, 2)
        dataset['regist_avg_aheadMonth'] = self.calcAvgShift(dataset.regist, 31, 2)

        dataset['casual_inc_ratio_monthly'] = self.calcIncreaseRatio(self.calcAvgShift(dataset.casual, 31, 2),
                                                                     dataset.casual_lag2)
        dataset['regist_inc_ratio_monthly'] = self.calcIncreaseRatio(self.calcAvgShift(dataset.regist, 31, 2),
                                                                     dataset.regist_lag2)

        dataset['casual_inc_ratio_weekly'] = self.calcIncreaseRatio(self.calcAvgShift(dataset.casual, 7, 2),
                                                                    dataset.casual_lag2)
        dataset['regist_inc_ratio_weekly'] = self.calcIncreaseRatio(self.calcAvgShift(dataset.regist, 7, 2),
                                                                    dataset.regist_lag2)
        dataset['casual_lastWeekday'] = dataset.casual.shift(7)
        dataset['regist_lastWeekday'] = dataset.regist.shift(7)

        dataset['casual_max_LastWeek'] = dataset.casual.shift(2).rolling(7, 1).max()
        dataset['regist_max_LastWeek'] = dataset.casual.shift(2).rolling(7, 1).max()

        dataset['casual_inc_ratio'] = ((dataset['casual'] - dataset['casual_lag2']) / dataset['casual_lag2']).apply(
            lambda x: round(x, 3))
        dataset['regist_inc_ratio'] = ((dataset['regist'] - dataset['regist_lag2']) / dataset['regist_lag2']).apply(
            lambda x: round(x, 3))

        dataset.to_csv("./totalBikeSharing.csv", index=False)
        return dataset


def main():
    process = Preprocessing()
    df = process.load_data()
    df = process.verboseClean(df)
    df = process.featureEngineering(df)
    print(df)


if __name__ == '__main__':
    main()
