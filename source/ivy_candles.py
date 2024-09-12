"""Updater for Infinitus Vigilantis"""
import logging
import os
import pandas as pd
import source.ivy_commons as icy
from alpaca.data.enums import Adjustment
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime
from queue import Queue
from multiprocessing import Queue as mpQueue
from os import path, listdir, cpu_count, remove, mkdir
from os.path import abspath, dirname, exists, join, realpath
from source.ivy_cartography import cartography
from source.ivy_watchlist import ivy_watchlist
__author__ = 'Daniel Ward'
__copyright__ = 'Copyright 2024, Daniel Ward'
__license__ = 'GPL v3'
__version__ = 'gardneri'


class Candelabrum:
    """Handler for historical price data."""
    def __init__(self, spawn_workers=False, ftype='thread'):
        """Start worker threads/processes and create folders if needed."""
        self.spawn_workers = spawn_workers
        ROOT_PATH = abspath(join(dirname(realpath(__file__)), '..'))
        self.CAULDRON_PATH = join(ROOT_PATH, 'cauldron')
        self.LOG_PATH = LOG_PATH = abspath(join(ROOT_PATH, 'logs'))
        self.LOG_FILE = LOG_FILE = join(LOG_PATH, 'ivy_candelabrum.log')
        self.DATA_PATH = DATA_PATH = abspath(join(ROOT_PATH, 'candelabrum'))
        self.DATA_FILE = join(DATA_PATH, '{}.ivy')
        self.DATA_BENCHMARKS = join(DATA_PATH, 'candelabrum.benchmarks')
        self.DATA_CANDLES = join(DATA_PATH, 'candelabrum.candles')
        self.DATA_FEATURES = join(DATA_PATH, 'candelabrum.features')
        self.DATA_SYMBOLS = join(DATA_PATH, 'candelabrum.symbols')
        for _path in (self.CAULDRON_PATH, self.LOG_PATH, self.DATA_PATH):
            if not exists(_path):
                mkdir(_path)
        if exists(LOG_FILE):
            remove(LOG_FILE)
        logging.getLogger('asyncio').setLevel(logging.ERROR)
        logging.basicConfig(
            filename=LOG_FILE,
            encoding='utf-8',
            level=logging.ERROR,
        )
        if spawn_workers:
            dispatcher = icy.ivy_dispatcher
            self._FTYPE = str(ftype)
            self._WORKERS = list()
            self._CPU_COUNT = cpu_count()
            self._MAX_THREADS = 3
            if self._FTYPE == 'process':
                self._QUEUE = mpQueue()
                logging.info(f'creating {self._CPU_COUNT - 1} processes...')
                r = range(self._CPU_COUNT - 1)
            else:
                self._QUEUE = Queue()
                logging.info(f'creating {self._MAX_THREADS} threads...')
                r = range(self._MAX_THREADS)
            for _ in r:
                self._WORKERS.append(dispatcher(self.__worker__, ftype=ftype))
        logging.info('Candelabrum: initialized.')

    def __worker__(self):
        """Get jobs and do work."""
        #chart_path = path.abspath(self._CHART_PATH)
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
            except Exception as details:
                logging.error('\n'.join(*details))
                continue

    def join_workers(self):
        """Block until all jobs are finished."""
        logging.info('waiting for all jobs to finish...')
        for _ in self._WORKERS:
            self._QUEUE.put('exit')
        if self._FTYPE == 'process':
            for w in self._WORKERS:
                w.join()
        else:
            self._QUEUE.join()

    def build_candles(self):
        ALPACA_ID = os.environ["APCA_API_KEY_ID"]
        if not len(ALPACA_ID) > 0:
            logging.error('Error: ALPACA_ID required.')
            return None
        ALPACA_SECRET = os.environ["APCA_API_SECRET_KEY"]
        if not len(ALPACA_SECRET) > 0:
            logging.error('Error: ALPACA_SECRET required.')
            return None
        file_path = self.DATA_FILE
        params_extras = dict(
            start=datetime(2016, 1, 1),
            end=datetime(*datetime.today().timetuple()[:3]),
            limit=10000,
            timeframe=TimeFrame.Day,
            adjustment=Adjustment('all'),
            )
        watchlist = list(ivy_watchlist)
        n_glob = 3
        shepherd = StockHistoricalDataClient(ALPACA_ID, ALPACA_SECRET)
        while len(watchlist) > 0:
            symbols = watchlist[:n_glob]
            request_params = StockBarsRequest(
                symbol_or_symbols=symbols,
                **params_extras,
                )
            wax_glob = shepherd.get_stock_bars(request_params).df
            print(symbols, wax_glob.shape)
            for symbol in symbols:
                candle_path = file_path.format(symbol)
                wax_glob.xs(symbol, level=0).to_csv(candle_path, mode='w+')
            del(watchlist[:n_glob])

    def omenize(self, symbol, reverse_dataset=False, n_trim=34):
        get_indicators = icy.get_indicators
        dataset = pd.read_csv(self.DATA_FILE.format(symbol))
        if len(dataset) <= n_trim * 2:
            return list()
        if reverse_dataset:
            dataset = dataset.reindex(index=dataset.index[::-1])
        omens = get_indicators(dataset)
        candles = dataset.merge(omens, left_index=True, right_index=True)
        candles.set_index('timestamp', inplace=True)
        return candles.iloc[n_trim:].copy()

    def light_candles(self):
        from torch import cuda, device, tensor
        from torch import float as tfloat
        omenize = self.omenize
        dev = device('cuda:0' if cuda.is_available() else 'cpu')
        cdl_args = dict(device=dev, dtype=tfloat)
        tensorize = lambda df: tensor(df.to_numpy(), **cdl_args)
        lit_candles = {}
        for symbol in ivy_watchlist:
            if symbol not in ('QQQ', 'SPY'):
                print(f'{symbol}: lighting candles...')
                candles = omenize(symbol, reverse_dataset=False)
                if len(candles) == 0:
                    continue
                lit_candles[symbol] = tensorize(candles)
        return lit_candles

    def get_benchmarks(self):
        from torch import cat, cuda, device, tensor
        from torch import float as tfloat
        omenize = self.omenize
        dev = device('cuda:0' if cuda.is_available() else 'cpu')
        cdl_args = dict(device=dev, dtype=tfloat)
        tensorize = lambda df: tensor(df.to_numpy(), **cdl_args)
        candles = (
            omenize('QQQ', reverse_dataset=True),
            omenize('QQQ', reverse_dataset=False),
            omenize('SPY', reverse_dataset=True),
            omenize('SPY', reverse_dataset=False),
            )
        features = list(candles[0].iloc[-1].index)
        benchmarks = cat([tensorize(cdl).unsqueeze(0) for cdl in candles])
        return (features, benchmarks)

    def rotate(self):
        from pickle import dump
        from torch import save
        features, benchmarks = self.get_benchmarks()
        with open(self.DATA_FEATURES, 'wb+') as file_obj:
            dump(features, file_obj)
        save(benchmarks, self.DATA_BENCHMARKS)
        lit_candles = self.light_candles()
        with open(self.DATA_SYMBOLS, 'wb+') as file_obj:
            dump(list(lit_candles.keys()), file_obj)
        save(lit_candles, self.DATA_CANDLES)

