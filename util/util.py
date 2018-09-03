__author__ = 'alexjzy'

from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns


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


def calc_rmse(data1, data2):
    mse = mean_squared_error(data1, data2)
    rmse = mse ** 0.5
    return rmse


def scatter_plot(input, x_col, y_col, model_name, rmse):
    g = sns.lmplot(x=x_col, y=y_col, data=input, scatter_kws={'s': 5})
    g.set_axis_labels("Predict", "Actual")
    plt.title('%s test prediction vs orginal, RMSE = %f' % (model_name, rmse), fontsize=10)
    g.savefig('./result/%s test prediction vs orginal, RMSE = %f (model_name, rmse).pdf' % (model_name, rmse))
    return g


def timeline_plot(timeline, data, title):
    colors = ['cornflowerblue', 'orange', 'green', 'purple']
    plt.figure(figsize=(12, 3))
    for i in range(len(data)):
        if i == 0:
            plt.plot(timeline, data[i], c=colors[i], label='originial')
        else:
            plt.plot(timeline, data[i], c=colors[i])
    plt.ylabel('Value', fontsize=10)
    plt.xlabel('Time', fontsize=10)
    plt.title(title)
    plt.savefig('./result/%s.pdf' % title, bbox_inches='tight')
    plt.legend(loc='upper left')
    plt.show()


def timeline_plot_overall(timeline, data, title):
    colors = ['cornflowerblue', 'orange', 'green', 'purple']
    plt.figure(figsize=(12, 3))
    labels = ['benchmark', "stacking", "blending"]
    for i in range(len(data)):
        if i == 0:
            plt.plot(timeline, data[i], c=colors[i], label='originial')
        else:
            plt.plot(timeline, data[i], c=colors[i], label=labels[i - 1])
    plt.ylabel('Value', fontsize=10)
    plt.xlabel('Time', fontsize=10)
    plt.title(title)
    plt.savefig('./result/%s.pdf' % title, bbox_inches='tight')
    plt.legend(loc='lower left')
    plt.show()
