import numpy as np
import backtrader as bt
import ib
import datetime
import os.path 
import sys
import numpy as np

class St(bt.Strategy):
    print("hello")
    def logdata(self):
        txt = []
        txt.append('{}'.format(len(self)))
           
        txt.append('{}'.format(
            self.data.datetime.datetime(0).isoformat())
        )
        txt.append('{:.2f}'.format(self.data.open[0]))
        txt.append('{:.2f}'.format(self.data.high[0]))
        txt.append('{:.2f}'.format(self.data.low[0]))
        txt.append('{:.2f}'.format(self.data.close[0]))
        txt.append('{:.2f}'.format(self.data.volume[0]))
        print(','.join(txt))
    
    data_live = False
    def notify_data(self, data, status, *args, **kwargs):
        print('*' * 5, 'DATA NOTIF:', data._getstatusname(status),
              *args)
        if status == data.LIVE:
            self.data_live = True

    def next(self):
        print("hello")
        # self.logdata()

def run(args=None):
    cerebro = bt.Cerebro(stdstats=False)
    print("2")
    ibstore = bt.stores.IBStore(host='127.0.0.1', port=7497, clientId=35)
    data = ibstore.getdata(dataname='EUR.USD-CASH-IDEALPRO') 
    print("3")
    cerebro.resampledata(data, timeframe=bt.TimeFrame.Seconds,
                         compression=10)
    print("4")
    cerebro.addstrategy(St)
    print("5")
    cerebro.run()
    print("6")
    cerebro.run()

if __name__ == '__main__':
    run()