"""Updater for Infinitus Vigilantis"""
import json
import numpy
import pandas as pd
import pickle
import random
import time
import traceback
import source.ivy_commons as icy
import source.ivy_alpaca as api
from os import path, listdir, cpu_count, remove, mkdir
from datetime import datetime
from queue import Queue
from multiprocessing import Queue as mpQueue
from source.ivy_cartography import cartography
#from source.ivy_mouse import ThreeBlindMice
from source.ivy_watchlist import ivy_watchlist
__author__ = 'Daniel Ward'
__copyright__ = 'Copyright 2024, Daniel Ward'
__license__ = 'GPL v3'
__version__ = 'gardneri'
SHEPHERD = api.AlpacaShepherd()
COLUMN_NAMES = {
    't': 'utc_ts',
    'o': 'open',
    'h': 'high',
    'l': 'low',
    'c': 'close',
    'v': 'volume',
    'n': 'num_trades',
    'vw': 'vol_wma_price'
    }


class Candelabrum:
    """Handler for historical price data."""
    def __init__(self, index=None, ftype='process'):
        """Set symbol index from composite index."""
        self._FTYPE = str(ftype)
        self._WORKERS = list()
        self._CPU_COUNT = cpu_count()
        self._MAX_THREADS = 3
        self._DATA_PATH = './candelabrum'
        self._ERROR_PATH = './errors'
        self._CHART_PATH = './charts'
        sym_path = [
            self._DATA_PATH,
            self._ERROR_PATH,
            self._CHART_PATH
            ]
        for _path in sym_path:
            if not path.exists(path.abspath(_path)):
                mkdir(_path)
        self._PREFIX = 'Candelabrum:'
        self._tz = 'America/New_York'
        p = lambda c: pd.to_datetime(c, utc=True).tz_convert(self._tz)
        if self._FTYPE == 'process':
            self._QUEUE = mpQueue()
            print(self._PREFIX, f'creating {self._CPU_COUNT - 1} processes...')
            r = range(self._CPU_COUNT - 1)
        else:
            self._QUEUE = Queue()
            print(self._PREFIX, f'creating {self._MAX_THREADS} threads...')
            r = range(self._MAX_THREADS)
        for _ in r:
            self._WORKERS.append(
                icy.ivy_dispatcher(
                    self.__worker__,
                    ftype=ftype
                    )
                )
        print('Candelabrum: initialized.')

    def __worker__(self):
        """Get jobs and do work."""
        chart_path = path.abspath(self._CHART_PATH)
        while True:
            job = self._QUEUE.get()
            if job == 'exit':
                break
            try:
                if job[0] == 'cartography':
                    chart_symbol = job[1]
                    candelabrum_candles = job[2]
                    cheese = job[3]
                    if len(job) == 5:
                        c_path = job[4]
                    else:
                        c_path = f'{chart_path}/{chart_symbol}.png'
                    cartography(
                        str(chart_symbol),
                        candelabrum_candles,
                        cheese=cheese,
                        chart_path=c_path,
                        chart_size=365,
                        )
            except Exception as err:
                err_path = f'./errors/{time.time()}-worker.exception'
                err_msg = f'{type(err)}:{err.args}\n\n{job}'
                with open(path.abspath(err_path), 'w+') as err_file:
                    err_file.write(err_msg)
                print(self._PREFIX, f'Worker Thread: {err_msg}')
                traceback.print_exc()
                continue

    def join_workers(self):
        """Block until all jobs are finished."""
        print(self._PREFIX, 'waiting for all jobs to finish...')
        for _ in self._WORKERS:
            self._QUEUE.put('exit')
        if self._FTYPE == 'process':
            for w in self._WORKERS:
                w.join()
        else:
            self._QUEUE.join()

def candle_maker(candles):
    """Makes a candle."""
    if len(candles) > 0 and type(candles) == pd.Series:
        name = str(candles.name)
        value = None
        if name == 'open':
            value = float(candles[0])
        elif name == 'high':
            value = float(candles.max())
        elif name == 'low':
            value = float(candles.min())
        elif name == 'close':
            value = float(candles[-1])
        elif name == 'volume':
            value = float(candles.sum())
        elif name == 'vol_wma_price':
            value = float(candles.mean())
        return value


def build_historical_database(start_date='2018-01-01'):
    """Create and light daily candles from historical data."""
    from time import strftime, strptime
    from torch import cuda, device, save, stack, tensor
    from torch import float as tfloat
    msg = 'Build Historical Database: {}'
    candelabrum_path = path.abspath('./candelabrum')
    if not path.exists(candelabrum_path):
        mkdir(candelabrum_path)
    dev = device('cuda:0' if cuda.is_available() else 'cpu')
    today = strftime('%Y-%m-%d', time.localtime())
    shepherd_args = dict(
        adjustment='all',
        timeframe='1Day',
        start=start_date,
        limit='10000',
        )
    tz = 'America/New_York'
    ts = pd.Timestamp
    date_args = dict(name='time')
    date_args['start'] = start_date
    date_args['end'] = today
    date_args['tz'] = tz
    date_args['freq'] = '1D'
    date_range = pd.date_range(**date_args)
    print(msg.format('Requesting calendar...'))
    calendar = list()
    market_days = list()
    for market_session in SHEPHERD.calendar():
        calendar.append(market_session['date'])
    for index, date in enumerate(date_range):
        date = str(date)[:10]
        if date in calendar:
            market_days.append(index)
    candles = dict()
    print(msg.format('Fetching data...'))
    for symbol in ivy_watchlist:
        try:
            data = SHEPHERD.candles(symbol, **shepherd_args)
            bars = pd.DataFrame(data['bars'])
            bars['time'] = [ts(t, unit='s', tz=tz) for t in bars['t']]
            bars.set_index('time', inplace=True)
            bars.rename(columns=COLUMN_NAMES, inplace=True)
            candles[symbol] = bars.copy()
        except Exception as details:
            print(f'{symbol}:', type(details), details.args)
        finally:
            continue
    p = candelabrum_path + '/{}'
    candelabrum = list()
    omenize = icy.get_indicators
    print(msg.format(f'Applying indicators to {len(candles.keys())} symbols.'))
    for symbol in candles.keys():
        del(candles[symbol]['utc_ts'])
        del(candles[symbol]['num_trades'])
        cdls = pd.DataFrame(index=date_range, columns=candles[symbol].keys())
        cdls.update(candles[symbol])
        cdls.fillna(method='ffill', inplace=True)
        cdls.fillna(method='bfill', inplace=True)
        cdls.dropna(inplace=True)
        cdls = cdls.transpose()
        cdls.replace(to_replace=0, method='ffill', inplace=True)
        cdls = cdls.transpose()
        cdls.dropna(inplace=True)
        cdls = cdls.resample('1D').apply(candle_maker)
        cdls.dropna(inplace=True)
        cdls = cdls.merge(omenize(cdls), left_index=True, right_index=True)
        cdls.to_csv(p.format(f'{symbol}.ivy'), mode='w+')
        candelabrum.append(tensor(cdls.to_numpy(), device=dev, dtype=tfloat))
    candelabrum = stack(candelabrum).transpose(0, 1)
    candelabrum = candelabrum[market_days]
    save(candelabrum, p.format('candelabrum.candles'))
    with open(path.abspath('./candelabrum/candelabrum.symbols'), 'wb+') as f:
        pickle.dump(list(candles.keys()), f)
    with open(path.abspath('./candelabrum/candelabrum.features'), 'wb+') as f:
        pickle.dump(list(cdls.keys()), f)
    m = 'Candelabrum created with {} candles for {} symbols with {} features.'
    print(msg.format(m.format(*candelabrum.shape)))
