__author__ = 'alexjzy'

import pandas as pd


def restore_ratio_result(pred, raw_lag):
    return (pred + 1) * raw_lag


def construct_result(raw_data, raw_lag, pred, target):
    revenue = 3
    loan = 2
    result = pd.DataFrame({
        'raw_Y': raw_data,
        'raw_lag': raw_lag,
        'target_Y_ratio': target,
        'pred_Y_ratio': pred,
        'pred_Y': restore_ratio_result(pred, raw_lag)}
    )
    result['cost'] = result.apply(
        lambda x: x['pred_Y'] * loan, axis=1
    )
    result['revenue'] = result.apply(
        lambda x: revenue * min(x['pred_Y'], x['raw_Y']), axis=1
    )
    result['profit'] = result['revenue'] - result['cost']
    result['benchmark_profit'] = result.apply(
        lambda x: (min(x['raw_lag'], x['raw_Y']) * revenue - loan * x['raw_lag']),
        axis=1
    )
    profit_sum = result.profit.sum()
    return result, profit_sum


if __name__ == '__main__':
    pass
