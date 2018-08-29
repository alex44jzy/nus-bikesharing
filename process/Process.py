__author__ = 'alexjzy'
import sys
import glob
import os

sys.path.append('../')
import xgboost
from model.NeuralNetwork import NeuralNetwork
from model.RandomForest import RandomForest
from model.Xgboost import Xgboost
from model.Stacking import StackingAveragedModels
import process.Preprocessing as preprocess
import profit.Profit as profitCalculator
from keras.models import load_model
from sklearn.externals import joblib
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import matplotlib.pyplot as plt
from util.Util import calc_rmse, scatter_plot, timeline_plot


def nn_prediction(model_existed=True):
    # get the 1 year train and test dataset
    if model_existed:
        filename = glob.glob(r'./*.h5')[0]
        nn_net = load_model(filename)
    else:
        model = NeuralNetwork(20, 15, 0.25)
        nn_net = model.construct(train_x, train_y, test_x, test_y)

    features.append(train_x.columns.values.tolist())
    pred_train = nn_net.predict(train_x)
    pred_test = nn_net.predict(test_x)

    result, summary = profitCalculator.construct_result(test_y_origin, test_raw.cnt_lag2, pred_test.flatten(), test_y)
    result.to_csv("./result/neural_network_%i-%i-%i.csv" % (year, month, day))
    print("neural network result:")
    print("predict rmse: %f" % calc_rmse(result.raw_Y, result.pred_Y))
    print("ratio rmse: %f" % calc_rmse(result.target_Y_ratio, result.pred_Y_ratio))
    print("profit: %f" % summary)
    print("-------------------------------------------------")
    # scatter_plot(result, 'pred_Y_ratio', 'target_Y_ratio', 'Neural Network',
    #              calc_rmse(result.target_Y_ratio, result.pred_Y_ratio))
    #
    # scatter_plot(result, 'pred_Y', 'raw_Y', 'Neural Network',
    #              calc_rmse(result.pred_Y, result.raw_Y))
    # timeline_plot(test_raw.dteday, [result.raw_Y, result.pred_Y], 'Neural network test prediction vs actual')
    # timeline_plot(test_raw.dteday, [abs(result.raw_Y - result.pred_Y)], 'Neural network absolute error through timeline')

    return nn_net, result, summary


def save_model():
    model, res, summ = nn_prediction(False)
    while summ < 1670000:
        model, res, summ = nn_prediction(False)
    model.save('nn_model-%i.h5' % summ)
    del model


def rf_prediction():
    cols = train_x.columns.values.tolist()
    lt = ['workingday',
          'weathersit_',
          'weekday_',
          'season_',
          'cnt_inc_ratio_lag2',
          'cnt_inc_ratio_monthly_avg',
          'cnt_inc_ratio_weekly_avg',
          'cnt_inc_ratio_weekly_max',
          'cnt_inc_ratio_weekly_min',
          'temp_inc']

    rf_cols = [x for x in cols if any(xs in x for xs in lt)]
    train_x_rf = train_x[rf_cols]
    test_x_rf = test_x[rf_cols]
    rf_model = RandomForest(train_x_rf, train_y)
    pred = rf_model.predict(test_x_rf)
    result, summary = profitCalculator.construct_result(test_y_origin, test_raw.cnt_lag2, pred, test_y)

    result.to_csv('./test_rf_1halfyear_bal.csv')


    features.append(train_x.columns.values.tolist())
    joblib.dump(rf_model, './rf_model.pkl')
    result.to_csv("./result/random_forest_%i-%i-%i.csv" % (year, month, day))
    print("random forest result:")
    print("predict rmse: %f" % calc_rmse(result.raw_Y, result.pred_Y))
    print("ratio rmse: %f" % calc_rmse(result.target_Y_ratio, result.pred_Y_ratio))
    print("profit: %f" % summary)
    print("-------------------------------------------------")
    #
    # scatter_plot(result, 'pred_Y_ratio', 'target_Y_ratio', 'Random Forest',
    #              calc_rmse(result.target_Y_ratio, result.pred_Y_ratio))
    #
    # scatter_plot(result, 'pred_Y', 'raw_Y', 'Random Forest',
    #              calc_rmse(result.pred_Y, result.raw_Y))
    # timeline_plot(test_raw.dteday, [abs(result.raw_Y - result.pred_Y)], 'Random forest test prediction vs actual')

    return result


def xgb_prediction():
    xgb_cols = ['holiday', 'workingday', 'temp', 'hum', 'cnt_lag2', 'temp_inc',
                'cnt_avg_aheadWeek', 'cnt_avg_ahead3days', 'cnt_avg_aheadMonth',
                'cnt_median_LastWeek', 'cnt_inc_ratio_lag2',
                'cnt_inc_ratio_monthly_avg', 'cnt_inc_ratio_weekly_avg',
                'cnt_lastWeekday', 'season_1', 'season_2', 'season_3', 'season_4',
                'weathersit_1', 'weathersit_2', 'weathersit_3', 'weathersit_4']

    train_x_xgb = train_x[xgb_cols]
    test_x_xgb = test_x[xgb_cols]
    xgb_model = Xgboost(train_x_xgb, train_y)
    pred = xgb_model.predict(test_x_xgb)
    result, summary = profitCalculator.construct_result(test_y_origin, test_raw.cnt_lag2, pred, test_y)
    features.append(xgb_cols)
    joblib.dump(xgb_model, './xgb_model.pkl')
    result.to_csv("./result/xgboost_%i-%i-%i.csv" % (year, month, day))
    print("Xgboost result:")
    print("predict rmse: %f" % calc_rmse(result.raw_Y, result.pred_Y))
    print("ratio rmse: %f" % calc_rmse(result.target_Y_ratio, result.pred_Y_ratio))
    print("profit: %f" % summary)
    print("-------------------------------------------------")

    # scatter_plot(result, 'pred_Y_ratio', 'target_Y_ratio', 'Xgboost',
    #              calc_rmse(result.target_Y_ratio, result.pred_Y_ratio))
    # scatter_plot(result, 'pred_Y', 'raw_Y', 'Xgboost',
    #              calc_rmse(result.pred_Y, result.raw_Y))
    # timeline_plot(test_raw.dteday, [result.raw_Y, result.pred_Y], 'Xgboost test prediction vs actual')

    return result


def ensemble_stacking():
    xgb_param = {
        'n_estimators': 100,
        'learning_rate': 0.09,
        'gamma': 0.1,
        'subsample': 0.51,
        'colsample_bytree': 0.9,
        'max_depth': 8,
        'alpha': 1,
        'reg_lambda': 9
    }

    xgb_model = xgboost.XGBRegressor(**xgb_param)

    rf_param = {
        'n_estimators': 100,
        'max_features': 'sqrt',
        'max_depth': 16,
    }
    rf_model = RandomForestRegressor(**rf_param)
    nn_model = load_model('./nn_model-1668217.h5')

    meta_model = xgboost.XGBRegressor()

    base_model = (nn_model, rf_model, xgb_model)
    ensemble = StackingAveragedModels(base_model, meta_model, 5)
    ensemble.fit(train_x, train_y, features)
    ensemble_pred = ensemble.predict(test_x, features)
    result, summary = profitCalculator.construct_result(test_y_origin, test_raw.cnt_lag2, ensemble_pred, test_y)
    result.to_csv("./result/stacking_%i-%i-%i.csv" % (year, month, day))
    print("stacking result:")
    print("predict rmse: %f" % calc_rmse(result.raw_Y, result.pred_Y))
    print("ratio rmse: %f" % calc_rmse(result.target_Y_ratio, result.pred_Y_ratio))
    print("profit: %f" % summary)
    print("-------------------------------------------------")
    print(summary)
    scatter_plot(result, 'pred_Y_ratio', 'target_Y_ratio', 'Stacking',
                 calc_rmse(result.target_Y_ratio, result.pred_Y_ratio))
    scatter_plot(result, 'pred_Y', 'raw_Y', 'Stacking',
                 calc_rmse(result.pred_Y, result.raw_Y))
    timeline_plot(test_raw.dteday, [result.raw_Y, result.pred_Y], 'Stacking test prediction vs actual')



def blending():
    _, res_nn, _ = nn_prediction()
    res_rf = rf_prediction()
    res_xgb = xgb_prediction()
    rr = {}

    percent = [0.01 * i for i in range(0, 100)]

    for per in percent:
        revenue = 3
        loan = 2
        avg = per * res_xgb.pred_Y + (1 - per) * res_nn.pred_Y
        result = pd.DataFrame({'avg': avg, 'cnt': test_raw.cnt})
        result['prof'] = result.apply(
            lambda x: (min(x['avg'], x['cnt']) * revenue - loan * x['avg']),
            axis=1
        )
        rr[per] = result.prof.sum()

    param = max(rr, key=rr.get)
    optimal_pred = param * res_xgb.pred_Y_ratio + (1 - param) * res_nn.pred_Y_ratio
    print(param)
    result, summary = profitCalculator.construct_result(test_y_origin, test_raw.cnt_lag2, optimal_pred, test_y)
    result.to_csv("./result/blending_%i-%i-%i.csv" % (year, month, day))
    scatter_plot(result, 'pred_Y_ratio', 'target_Y_ratio', 'Blending',
                 calc_rmse(result.target_Y_ratio, result.pred_Y_ratio))
    scatter_plot(result, 'pred_Y', 'raw_Y', 'Blending',
                 calc_rmse(result.pred_Y, result.raw_Y))
    timeline_plot(test_raw.dteday, [result.raw_Y, result.pred_Y], 'Blending test prediction vs actual')

    # print("blending result:")
    # print("predict rmse: %f" % calc_rmse(result.raw_Y, result.pred_Y))
    # print("ratio rmse: %f" % calc_rmse(result.target_Y_ratio, result.pred_Y_ratio))
    # print("profit: %f" % summary)
    # print("-------------------------------------------------")



if __name__ == '__main__':
    year = 2012
    month = 1
    day = 1
    train_x, train_raw, train_y, train_y_origin, test_x, test_raw, test_y, test_y_origin = \
        preprocess.main(year, month, day, False)
    features = []
    nn_prediction()
    # rf_prediction()
    # xgb_prediction()
    # ensemble_stacking()
    # blending()
