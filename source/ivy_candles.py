"""Updater for Infinitus Vigilantis"""
import json
import numpy
import pandas
import pickle
import random
import time
import source.ivy_commons as icy
import source.ivy_alpaca as api
from source.ivy_mouse import ThreeBlindMice
from datetime import datetime
from os import path, listdir, cpu_count

__author__ = 'Daniel Ward'
__copyright__ = 'Copyright 2022, Daniel Ward'
__license__ = 'GPL v3'
SILENCE = icy.silence
DIV = icy.safe_div
PERCENT = icy.percent_change
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


def daemonize():
    """Keep spinning in the background while GUI is running."""
    return icy.ivy_dispatcher(spin_wheel, ftype='process')


def market_calendar(cpath='./indexes/calendar.index'):
    """Fetch the market calendar and store it locally."""
    if path.exists(cpath):
        ivy_calendar = pandas.read_csv(cpath, index_col=0, dtype=str)
    else:
        c = SHEPHERD.calendar()
        ivy_calendar = pandas.DataFrame(c, dtype=str)
        ivy_calendar.set_index('date', inplace=True)
        ivy_calendar.to_csv(cpath)
    return ivy_calendar.copy()


def alpaca_assets(asset_path='./indexes/alpaca.assets'):
    """Get assets, filter them, and store locally."""
    if path.exists(asset_path):
        with open(asset_path, 'r') as f:
            assets = json.loads(f.read())
    else:
        assets = dict()
        for obj in SHEPHERD.assets():
            symbol = obj['symbol']
            qualified = all((
                symbol.isalpha(),
                obj['status'] == 'active',
                obj['class'] == 'us_equity',
                obj['tradable'] == True
                ))
            if qualified:
                assets[symbol] = obj
        with open(asset_path, 'w+') as f:
            r = f.write(json.dumps(assets))
    return assets


def composite_index(ci_path='./indexes/composite.index'):
    """Validates assets and returns an index of symbols."""
    if path.exists(ci_path):
        ivy_ndx = pandas.read_csv(ci_path, index_col=0)
    else:
        valid_symbols = dict()
        assets = alpaca_assets()
        i = 0
        f = len(assets)
        print(f'Composite Index: {f} assets returned.')
        def validate_response(resp):
            """Ensure data exists for the entire range of dates."""
            validated = False
            try:
                k = resp['bars'][0].keys()
                b = COLUMN_NAMES.keys()
                validated = all([True if s in k else False for s in b])
            finally:
                return validated
        qa = dict(limit=10000)
        for symbol, v in assets.items():
            i += 1
            print(f'Composite Index: Validating {symbol} ({i}/{f})...')
            validated = [False, False]
            qa['start_date'] = '2019-01-02'
            qa['end_date'] = '2019-01-02'
            validated[0] = validate_response(SHEPHERD.candles(symbol, **qa))
            qa['start_date'] = '2022-01-03'
            qa['end_date'] = '2022-01-03'
            validated[1] = validate_response(SHEPHERD.candles(symbol, **qa))
            if all(validated):
                print(f'Composite Index: {symbol} validated!')
                valid_symbols[symbol] = v
        ivy_ndx = pandas.DataFrame()
        ivy_ndx['symbols'] = valid_symbols.keys()
        ivy_ndx['exchanges'] = valid_symbols.values()
        ivy_ndx.to_csv(ci_path)
    s = ivy_ndx['symbols'].tolist()
    e = ivy_ndx['exchanges'].tolist()
    if len(s) != len(e):
        return None
    print(f'Composite Index: {len(s)} valid assets.')
    return [(s[i], e[i]) for i in range(len(s))]


class Candelabrum:
    """Handler for historical price data."""
    def __init__(self, index=None, ftype='process'):
        """Set symbol index from composite index."""
        self.set_index = lambda n: [t[0] for t in n]
        if index:
            self._INDEX = self.set_index(index)
        else:
            self._INDEX = list()
        self._FTYPE = str(ftype)
        self._WORKERS = list()
        self._CPU_COUNT = cpu_count()
        self._MAX_THREADS = self._CPU_COUNT * 2 - 1
        self._BENCHMARKS = ('QQQ', 'SPY')
        self._DATA_PATH = './candelabrum'
        self._ERROR_PATH = './errors'
        self._IVI_PATH = './indicators'
        self._PREFIX = 'Candelabrum:'
        self._TIMER = icy.TimeKeeper()
        self.benchmarks = dict()
        self._tz = 'America/New_York'
        p = lambda c: pandas.to_datetime(c, utc=True).tz_convert(self._tz)
        self._CSV_ARGS = dict(
            index_col=0,
            parse_dates=True,
            infer_datetime_format=True,
            date_parser=p
            )
        if self._FTYPE == 'process':
            self._QUEUE = icy.mpQueue()
            print(self._PREFIX, f'creating {self._CPU_COUNT - 1} processes...')
            r = range(self._CPU_COUNT - 1)
        else:
            self._QUEUE = icy.Queue()
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

    def __get_path__(self, symbol, date_string):
        """You no take candle!"""
        p = f'{self._DATA_PATH}/{symbol.upper()}-{date_string}.ivy'
        return path.abspath(p)

    def __worker__(self):
        """Get jobs and save data locally."""
        while True:
            job = self._QUEUE.get()
            if job == 'exit':
                break
            try:
                if job[0] == 'indicators':
                    if not path.exists(job[1]):
                        with open(job[2]) as f:
                            candles = pandas.read_csv(f, **self._CSV_ARGS)
                        ivi = icy.get_indicators(candles)
                        ivi.to_csv(job[1])
                #elif job[0] == 'cartography':
                else:
                    data = json.loads(job[0])
                    bars = data['bars']
                    date_string = job[1]
                    requested_symbols = job[2]
                    returned_symbols = bars.keys()
                    tz = self._tz
                    pts = pandas.Timestamp
                    for symbol in returned_symbols:
                        df = pandas.DataFrame(bars[symbol])
                        df['time'] = [pts(t, unit='s', tz=tz) for t in df['t']]
                        df.set_index('time', inplace=True)
                        df.rename(columns=COLUMN_NAMES, inplace=True)
                        template = self.fill_template(df.copy())
                        file_name = f'{symbol.upper()}-{date_string}'
                        if len(template) > 0:
                            ivy_path = f'./candelabrum/{file_name}.ivy'
                            template.to_csv(path.abspath(ivy_path), mode='w+')
                        else:
                            err_path = f'./errors/{file_name}.error'
                            with open(path.abspath(err_path), 'w+') as err_file:
                                err_file.write('empty template')
                    for symbol in requested_symbols:
                        if symbol not in returned_symbols:
                            file_name = f'{symbol.upper()}-{date_string}'
                            err_path = f'./errors/{file_name}.error'
                            with open(path.abspath(err_path), 'w+') as err_file:
                                err_file.write('not in returned symbols')
            except Exception as err:
                err_path = f'./errors/{time.time()}-worker.exception'
                err_msg = f'{type(err)}:{err.args}\n\n{job}'
                with open(path.abspath(err_path), 'w+') as err_file:
                    err_file.write(err_msg)
                print(self._PREFIX, f'Worker Thread: {err_msg}')

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

    def fill_template(self, dataframe):
        """Populate template and interpolate missing data."""
        date_args = dict(name='time')
        date_args['start'] = dataframe.index[0]
        date_args['end'] = dataframe.index[-1]
        date_args['tz'] = self._tz
        date_args['freq'] = '1min'
        i = pandas.date_range(**date_args)
        c = COLUMN_NAMES.values()
        template = pandas.DataFrame(index=i, columns=c)
        template.update(dataframe)
        template.fillna(method='ffill', inplace=True)
        template.fillna(method='bfill', inplace=True)
        template.dropna(inplace=True)
        template = template.transpose().copy()
        template.replace(to_replace=0, method='ffill', inplace=True)
        template = template.transpose().copy()
        template.dropna(inplace=True)
        return template.copy()

    def load_candles(self, symbol, date_string):
        """=^.^="""
        data_path = self.__get_path__(symbol, date_string)
        if not path.exists(data_path):
            return pandas.DataFrame()
        with open(data_path) as pth:
            df = pandas.read_csv(pth, **self._CSV_ARGS)
        return df.copy()

    def save_candles(self, symbol, dataframe, date_string):
        """=^.^="""
        dataframe.to_csv(self.__get_path__(symbol, date_string), mode='w+')

    def apply_indicators(self, dataframe=None):
        """Save IVi file for every IVy file unless dataframe is provided."""
        if dataframe is None:
            data_path = path.abspath(self._DATA_PATH)
            ivy_files = listdir(data_path)
            total_files = len(ivy_files)
            i = 0
            for file_name in ivy_files:
                i += 1
                ivi_path = path.abspath(f'{self._IVI_PATH}/{file_name[:-4]}.ivi')
                ivy_path = path.abspath(f'{data_path}/{file_name}')
                self._QUEUE.put(('indicators', ivi_path, ivy_path))
            self.join_workers()
        else:
            return icy.get_indicators(dataframe)

    def do_update(self, symbols, **kwargs):
        """Update historical price data for a given symbol."""
        ts = kwargs['start']
        valid_symbols = list()
        for s in symbols:
            file_name = f'{s.upper()}-{ts}'
            ivy_path = f'{self._DATA_PATH}/{file_name}.ivy'
            err_path = f'{self._ERROR_PATH}/{file_name}.error'
            if not path.exists(path.abspath(ivy_path)):
                if not path.exists(path.abspath(err_path)):
                    valid_symbols.append(s)
        if not valid_symbols:
            return True
        print(self._PREFIX, f'{ts} updating {len(valid_symbols)} symbols.')
        gathering_data = True
        while gathering_data:
            q = SHEPHERD.candles(valid_symbols, **kwargs)
            if type(q) == str:
                self._QUEUE.put((q, ts, valid_symbols))
                token_str = '"next_page_token":'
                token_location = q.find(token_str, -100) + len(token_str)
                parsed_token = q[token_location:-1]
                if parsed_token != 'null':
                    kwargs['page_token'] = parsed_token[1:-1]
                else:
                    gathering_data = False
            else:
                gathering_data = False
        return True

    def gather_data(self, symbol, start_date, end_date, daily=False):
        calendar = market_calendar()
        calendar_dates = calendar.index.tolist()
        date_obj = lambda t: datetime.strptime(t, '%Y-%m-%d')
        dates = (date_obj(start_date), date_obj(end_date))
        candles = None
        for ts in calendar_dates:
            day_obj = date_obj(ts)
            if day_obj < dates[0]: continue
            if day_obj > dates[1]: break
            try:
                day_data = self.load_candles(symbol, ts)
                if daily and day_data is not None:
                    yield ts, day_data
                elif candles is None:
                    candles = day_data.copy()
                else:
                    candles = pandas.concat([candles, day_data])
            except Exception as details:
                print(self._PREFIX, f'Encountered {details}')
                continue
        if not daily:
            return candles.copy()

    def get_daily_candles(self, symbol, start_time=None, end_time=None):
        """Gets all of the data and transforms it into daily candles."""
        gather_data = self.gather_data
        make_candle = self.daily_candle
        if start_time is None:
            start_time = '2019-01-01'
        if end_time is None:
            tomorrow = time.localtime(time.time() + 86400)
            end_time = time.strftime('%Y-%m-%d', tomorrow)
        candles = list()
        timestamps = list()
        daily_data = gather_data(symbol, start_time, end_time, daily=True)
        for ts, day_data in daily_data:
            candle = make_candle(day_data)
            if candle is not None:
                candles.append(candle)
                timestamps.append(ts)
        return pandas.DataFrame(candles, index=timestamps)

    def daily_candle(self, day_data):
        """Takes minute data and converts it into a daily candle."""
        if len(day_data) == 0: return None
        ohlc = day_data[['open', 'high', 'low', 'close']]
        return {
            'open': float(ohlc['open'][0]),
            'high': float(max(ohlc.max())),
            'low': float(min(ohlc.min())),
            'close': float(ohlc['close'][-1]),
            'volume': float(day_data['volume'].sum()),
            'num_trades': float(day_data['num_trades'].sum()),
            'vol_wma_price': float(day_data['vol_wma_price'].mean()),
            }

    def research_candles(self):
        get_daily = self.get_daily_candles
        omenize = self.apply_indicators
        moirai = ThreeBlindMice(34, 4, verbose=True)
        print(self._PREFIX, 'Starting research loop...')
        symbols = [s for s, e in composite_index()]
        symbols_researched = 0
        symbols_total = len(symbols)
        symbols_remaining = symbols_total
        epoch_msg = self._PREFIX + ' Sent {} to The Moirai. ( {} / {} ) '
        loop_start = time.time()
        while symbols_remaining > 0:
            symbols_researched += 1
            index = random.randint(0, symbols_remaining)
            symbol = symbols[index]
            symbols.pop(index)
            bars = get_daily(symbol)
            indicators = omenize(bars)
            candles = bars.merge(indicators, left_index=True, right_index=True)
            print(epoch_msg.format(symbol, symbols_researched, symbols_total))
            moirai.research(symbol, candles)
            pred = moirai.predictions[symbol]
            prefix = f'{self._PREFIX} {symbol}'
            print(prefix, 'proj_accuracy:', pred['proj_accuracy'])
            print(prefix, 'last_price:', pred['last_price'])
            print(prefix, 'proj_close:', pred['sealed_candles'][-1][-1].item())
            print(prefix, 'proj_gain:', pred['proj_gain'])
            symbols_remaining = len(symbols)
        elapsed = time.time() - loop_start
        message = 'Research of {} complete after {}.'
        if elapsed > 86400:
            message += '{} days'.format(round(elapsed / 86400, 5))
        elif elapsed > 3600:
            message += '{} hours'.format(round(elapsed / 3600, 5))
        elif elapsed > 60:
            message += '{} minutes'.format(round(elapsed / 60, 5))
        else:
            message += '{} seconds'.format(round(elapsed, 5))
        print(self._PREFIX, message.format(symbols_total, elapsed))

    def candle_maker(self, candles):
        if len(candles) > 0 and type(candles) == pandas.Series:
            name = str(candles.name)
            value = None
            if name == 'utc_ts':
                value = float(make_utc(candles[0]))
            elif name == 'open':
                value = float(candles[0])
            elif name == 'high':
                value = float(candles.max())
            elif name == 'low':
                value = float(candles.min())
            elif name == 'close':
                value = float(candles[-1])
            elif name == 'volume':
                value = float(candles.sum())
            return value
        return None

    def resample_candles(self, candles, scale):
        c = candles.resample(scale).apply(self.candle_maker)
        c.dropna(inplace=True)
        return c.copy()


def make_utc(time_string):
    """Liberation from the chains of Daylight Savings."""
    fmt = '%Y-%m-%dT%H:%M:%S%z'
    first_annoyance = time.strptime(time_string, fmt)
    second_annoyance = time.mktime(first_annoyance)
    third_annoyance = time.gmtime(second_annoyance)
    return time.mktime(third_annoyance)


def build_historical_database(starting_year=2019):
    """Cycle through calendar and update data accordingly."""
    calendar = market_calendar()
    calendar_dates = calendar.index.tolist()
    ivy_ndx = composite_index()
    today = time.strftime('%Y-%m-%d', time.localtime())
    local_year = int(today[0:4])
    local_month = int(today[5:7])
    local_day = int(today[8:])
    cdlm = Candelabrum()
    uargs = dict(limit=10000)
    keeper = icy.TimeKeeper()
    msg = 'Build Historical Database: {}'
    batch_limit = 224
    print(msg.format(f'BATCH_LIMIT={batch_limit}'))
    print(msg.format(f'INDEX_LENGTH={len(ivy_ndx)}'))
    print(msg.format(f'Checking calendar dates for missing data.'))
    for ts in calendar_dates:
        ts_year = int(ts[0:4])
        if ts_year < starting_year:
            continue
        ts_month = int(ts[5:7])
        ts_day = int(ts[8:])
        time_stop = (
            ts_year == local_year,
            ts_month == local_month,
            ts_day >= local_day
            )
        if all(time_stop):
            break
        uargs['start'] = str(ts)
        uargs['end'] = str(ts)
        symbols = list()
        batch_count = 0
        for s in ivy_ndx:
            symbols.append(s[0])
            batch_count += 1
            if batch_count == batch_limit:
                cdlm.do_update(symbols, **uargs)
                symbols = list()
                batch_count = 0
        if batch_count > 0:
            cdlm.do_update(symbols, **uargs)
    cdlm.join_workers()
    print(msg.format(f'completed after {keeper.final}.'))
