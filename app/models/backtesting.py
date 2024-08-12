# app/models/backtesting.py
import backtrader as bt
import pandas as pd

class TestStrategy(bt.Strategy):
    def __init__(self):
        self.dataclose = self.datas[0].close

    def next(self):
        if not self.position:  # Si nous ne sommes pas en position
            if self.dataclose[0] < self.dataclose[-1]:
                if self.dataclose[-1] < self.dataclose[-2]:
                    self.buy()
        else:
            if self.dataclose[0] > self.dataclose[-1]:
                self.sell()

def run_backtest(data_path):
    cerebro = bt.Cerebro()
    cerebro.addstrategy(TestStrategy)

    # Charger les données de marché historiques
    data = pd.read_csv(data_path, index_col='Date', parse_dates=True)
    datafeed = bt.feeds.PandasData(dataname=data)
    cerebro.adddata(datafeed)

    cerebro.broker.setcash(100000)
    cerebro.broker.setcommission(commission=0.001)

    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    cerebro.run()
    print('Ending Portfolio Value: %.2f' % cerebro.broker.getvalue())
    cerebro.plot()