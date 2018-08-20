__author__ = 'alexjzy'


class Util:
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
