__author__ = 'alexjzy'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler


def load_data():
    filename = '/day.csv'
    print('Start loading dataset')
    data_dir = '/dataset'
    os.chdir('./..')
    cwd = os.getcwd() + data_dir
    print(cwd)
    data_file_name = filename
    dataset = pd.read_csv(cwd + data_file_name)
    return dataset


def calcAvgShift(dfCol, period, exclude):
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


def calcMaxShift(dfCol, period, exclude):
    try:
        summary = dfCol.rolling(period).max()
        if period is None:
            return dfCol
        if exclude is None:
            return summary
        maximum = dfCol.shift(exclude).rolling(period, 1).max()
        return maximum.apply(lambda x: round(x, 3))
    except Exception as e:
        print("error:" + str(e))


def calcMinShift(dfCol, period, exclude):
    try:
        summary = dfCol.rolling(period).min()
        if period is None:
            return dfCol
        if exclude is None:
            return summary
        minimum = dfCol.shift(exclude).rolling(period, 1).min()
        return minimum.apply(lambda x: round(x, 3))
    except Exception as e:
        print("error:" + str(e))


def calcIncreaseRatio(pre, comp):
    inc = (comp - pre) / pre
    return round(inc, 3)


def verboseClean(dataset):
    # convert the data time type
    dataset.dteday = dataset.dteday.apply(lambda x: pd.to_datetime(x))
    dataset = dataset.drop(['instant'], axis=1)
    dataset = dataset.rename(columns={'registered': 'regist'})

    # remove the extreme weather date
    dataset = dataset.drop(dataset[dataset.dteday == '2012-10-29'].index)

    # manually modify some dates weathersit from level 3 to level 4
    amendWeathersit3_4 = ['2011-01-26', '2011-04-16', '2011-09-07', '2011-09-08',
                          '2011-10-29', '2011-12-07', '2012-10-29',
                          '2012-10-30', '2012-12-26']
    amendWeathersit2_3 = ['2012-04-22']
    dataset.loc[dataset.dteday.isin(amendWeathersit3_4), 'weathersit'] = 4
    dataset.loc[dataset.dteday.isin(amendWeathersit2_3), 'weathersit'] = 3
    return dataset


def feature_engineering(dataset):
    dataset['casual_lag2'] = dataset['casual'].shift(2)
    dataset['regist_lag2'] = dataset['regist'].shift(2)
    dataset['cnt_lag2'] = dataset['cnt'].shift(2)
    dataset['temp_inc'] = calcIncreaseRatio(dataset.temp.shift(1), dataset.temp)

    dataset['casual_avg_aheadWeek'] = calcAvgShift(dataset.casual, 7, 2)
    dataset['regist_avg_aheadWeek'] = calcAvgShift(dataset.regist, 7, 2)
    dataset['cnt_avg_aheadWeek'] = calcAvgShift(dataset.cnt, 7, 2)

    dataset['casual_avg_ahead3days'] = calcAvgShift(dataset.casual, 5, 2)
    dataset['regist_avg_ahead3days'] = calcAvgShift(dataset.regist, 5, 2)
    dataset['cnt_avg_ahead3days'] = calcAvgShift(dataset.cnt, 5, 2)

    dataset['casual_avg_aheadMonth'] = calcAvgShift(dataset.casual, 31, 2)
    dataset['regist_avg_aheadMonth'] = calcAvgShift(dataset.regist, 31, 2)
    dataset['cnt_avg_aheadMonth'] = calcAvgShift(dataset.cnt, 31, 2)

    dataset['casual_median_LastWeek'] = dataset.casual.shift(2).rolling(7, 1).median()
    dataset['regist_median_LastWeek'] = dataset.regist.shift(2).rolling(7, 1).median()
    dataset['cnt_median_LastWeek'] = dataset.cnt.shift(2).rolling(7, 1).median()

    dataset['casual_inc_ratio_lag2'] = calcIncreaseRatio(dataset.casual.shift(3), dataset.casual.shift(2))
    dataset['regist_inc_ratio_lag2'] = calcIncreaseRatio(dataset.regist.shift(3), dataset.regist.shift(2))
    dataset['cnt_inc_ratio_lag2'] = calcIncreaseRatio(dataset.cnt.shift(3), dataset.cnt.shift(2))

    dataset['casual_inc_ratio_weekly_max'] = calcIncreaseRatio(calcMaxShift(dataset.casual, 6, 3), dataset.casual_lag2)
    dataset['regist_inc_ratio_weekly_max'] = calcIncreaseRatio(calcMaxShift(dataset.regist, 6, 3), dataset.regist_lag2)
    dataset['cnt_inc_ratio_weekly_max'] = calcIncreaseRatio(calcMaxShift(dataset.cnt, 6, 3), dataset.cnt_lag2)

    dataset['casual_inc_ratio_weekly_min'] = calcIncreaseRatio(calcMinShift(dataset.casual, 6, 3), dataset.casual_lag2)
    dataset['regist_inc_ratio_weekly_min'] = calcIncreaseRatio(calcMinShift(dataset.regist, 6, 3), dataset.regist_lag2)
    dataset['cnt_inc_ratio_weekly_min'] = calcIncreaseRatio(calcMinShift(dataset.cnt, 6, 3), dataset.cnt_lag2)

    dataset['casual_inc_ratio_monthly_avg'] = calcIncreaseRatio(calcAvgShift(dataset.casual, 31, 2),
                                                                dataset.casual_lag2)
    dataset['regist_inc_ratio_monthly_avg'] = calcIncreaseRatio(calcAvgShift(dataset.regist, 31, 2),
                                                                dataset.regist_lag2)
    dataset['cnt_inc_ratio_monthly_avg'] = calcIncreaseRatio(calcAvgShift(dataset.cnt, 31, 2), dataset.cnt_lag2)

    dataset['casual_inc_ratio_weekly_avg'] = calcIncreaseRatio(calcAvgShift(dataset.casual, 7, 2), dataset.casual_lag2)
    dataset['regist_inc_ratio_weekly_avg'] = calcIncreaseRatio(calcAvgShift(dataset.regist, 7, 2), dataset.regist_lag2)
    dataset['cnt_inc_ratio_weekly_avg'] = calcIncreaseRatio(calcAvgShift(dataset.cnt, 7, 2), dataset.cnt_lag2)

    # xgboost 重要变量
    dataset['casual_lastWeekday'] = dataset.casual.shift(7)
    dataset['regist_lastWeekday'] = dataset.regist.shift(7)
    dataset['cnt_lastWeekday'] = dataset.cnt.shift(7)

    dataset['casual_inc_ratio'] = ((dataset['casual'] - dataset['casual_lag2']) / dataset['casual_lag2']).apply(
        lambda x: round(x, 3))
    dataset['regist_inc_ratio'] = ((dataset['regist'] - dataset['regist_lag2']) / dataset['regist_lag2']).apply(
        lambda x: round(x, 3))
    dataset['cnt_inc_ratio'] = ((dataset['cnt'] - dataset['cnt_lag2']) / dataset['cnt_lag2']).apply(
        lambda x: round(x, 3))

    dataset.to_csv("./totalBikeSharing.csv", index=False)
    return dataset


def dummy_variables(dataset):
    dummy_season = pd.get_dummies(dataset.season, prefix="season")
    dummy_weathersit = pd.get_dummies(dataset.weathersit, prefix="weathersit")
    dummy_weekday = pd.get_dummies(dataset.weekday, prefix="weekday")
    dataset = pd.concat([dataset, dummy_season, dummy_weathersit, dummy_weekday], axis=1)
    return dataset


def get_final_set(dataset, train, test, target='cnt'):
    target_col_name = target + '_inc_ratio'
    onehotcols = dataset.filter(regex='workingday|holiday|weekday_|weathersit_|season_').columns.values

    casual_cols = dataset.filter(like='casual').columns.values.tolist()  # filter the column name contain casual
    regist_cols = dataset.filter(like='regist').columns.values.tolist()
    cnt_cols = dataset.filter(like='cnt').columns.values.tolist()

    exclude_dict = {'casual': regist_cols + cnt_cols, 'regist': casual_cols + cnt_cols,
                    'cnt': casual_cols + regist_cols}
    drops = get_drop_columns_name(target, target_col_name, exclude_dict)

    train_y_origin = train[target]
    test_y_origin = test[target]

    train_y = train[target_col_name]
    test_y = test[target_col_name]

    train_x_afdrop = train.drop(drops, axis=1)
    test_x_afdrop = test.drop(drops, axis=1)

    nanindex, train_x, train_y = get_non_nan(train_x_afdrop, train_y)

    train_y_origin = train_y_origin.drop(nanindex, axis=0)

    standcols = [i for i in train_x_afdrop.columns.values if i not in onehotcols]
    train_x, test_x = standardization(standcols, train_x, test_x_afdrop)

    return train_x, train, train_y, train_y_origin, test_x, test, test_y, test_y_origin


def get_non_nan(x, y):
    nan_index = list(set(np.where(np.isnan(x))[0]))
    x = x.drop(nan_index, axis=0)
    y = y.drop(nan_index, axis=0)
    x = x.reset_index(drop=True)
    y = y.reset_index(drop=True)
    return nan_index, x, y


def standardization(cols, train, test):
    scaler = StandardScaler().fit(train[cols])
    train.loc[:, cols] = scaler.transform(train[cols])
    test.loc[:, cols] = scaler.transform(test[cols])
    return train, test


def get_train_test(dataset, year=2012, month=1, day=1):
    condition = dataset.dteday < datetime(year, month, day)

    train = dataset[condition]
    test = dataset[~condition]
    return train, test


def get_balance_train_test(dataset, year=2012, month=7, day=1, balance_start=(2011, 7, 1), balance_end=(2012, 1, 1)):
    train, test = get_train_test(dataset, year, month, day)

    cond1 = datetime(balance_end[0], balance_end[1], balance_end[2]) > dataset.dteday
    cond2 = datetime(balance_start[0], balance_start[1], balance_start[2]) <= dataset.dteday
    train_extra = dataset[cond1 & cond2]
    train_bal = pd.concat([train, train_extra], axis=0)
    return train_bal, test


def get_drop_columns_name(target, target_col, exclude):
    drop_col = ['dteday', 'weekday', 'yr', 'mnth',
                'season', 'weathersit', 'cnt', 'atemp', 'windspeed',
                ]
    drop_col.extend([target, target_col])
    drop_col.extend(list(exclude[target]))
    return drop_col


def main(year=2012, month=1, day=1, balance=False, target='cnt'):
    print(balance)
    df = load_data()
    df = verboseClean(df)
    df = feature_engineering(df)
    df = dummy_variables(df)

    if balance:
        train, test = get_balance_train_test(df)
    else:
        train, test = get_train_test(df, year, month, day)
    train_x, train_x_without_stad, train_y, train_y_origin, test_x, test_x_without_stad, test_y, test_y_origin = \
        get_final_set(df, train, test, target)
    return train_x, train_x_without_stad, train_y, train_y_origin, test_x, test_x_without_stad, test_y, test_y_origin


if __name__ == '__main__':
    main(False)
